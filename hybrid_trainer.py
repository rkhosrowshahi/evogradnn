

import argparse
import itertools
import os
import sys
import time

import jax
import numpy as np
import torch
from weight_sharing import GMMSoftWeightSharing, WeightSharingDecoder
from models import get_model
from utils import Logger, assign_flat_params, evaluate_model_acc, f1_score_error, fitness, flatten_params, freeze_bn, load_data, plot_convergence, set_seed, sgd_finetune, strategy_init, init_wandb, log_training_metrics, log_evolution_metrics, finish_wandb_run

def evaluate_solution(solution, ws, batch):
    return ws.objective(solution, batch)


def evaluate_solution_wrapper(args):
    solution, ws, batch = args
    return evaluate_solution(solution, ws, batch)


def gradfree_trainer(es, es_params, state, key, ws, train_loader, epoch):
    for countiter, batch in enumerate(train_loader):
        # Get new solutions
        key, key_ask, key_tell = jax.random.split(key, 3)
        solutions, state = es.ask(key_ask, state, es_params)
        
        # Evaluate solutions
        fitnesses = np.zeros(args.popsize)
        for i, z in enumerate(solutions):
            fitnesses[i] = ws.objective(centroids=z, batch=batch)
            # print(f"fitness of solution {i}: {fitnesses[i]}")
        # with Pool(processes=4) as pool:
        #     args = [(solution, ws, batch) for solution in solutions]
        #     fitnesses = pool.map(evaluate_solution_wrapper, args)
                
        # Update ES
        state, metrics = es.tell(key=key_tell, population=solutions, fitness=fitnesses, state=state, params=es_params)

        # Apply mean solution
        es_mean = es.get_mean(state=state)
        ws.update_base_theta(centroids=es_mean)
            
        # Log metrics
        print(f"Epoch {epoch}, iteration {countiter+1}, sigma: {np.mean(state.std):.6f}, "
                f"best fitness: {metrics['best_fitness']:.6f}, "
                f"best fitness in iteration: {metrics['best_fitness_in_generation']:.6f}, "
                f"avg fitness in iteration: {np.mean(fitnesses):.6e}, "
                f"best solution norm: {metrics['best_solution_norm']:.6f}, "
                f"mean norm: {metrics['mean_norm']:.6f}")
        
        # Log to wandb
        log_evolution_metrics(
            epoch=epoch,
            iteration=countiter+1,
            best_fitness=metrics['best_fitness'],
            best_fitness_in_generation=metrics['best_fitness_in_generation'],
            avg_fitness=np.mean(fitnesses),
            best_solution_norm=metrics['best_solution_norm'],
            additional_metrics={
                'sigma_mean': np.mean(state.std),
                'mean_norm': metrics['mean_norm']
            }
        )
        
        # if (countiter+1) % 10 == 0:
        #     break
        ws.sigma_eff2 = np.mean(state.std)
        
    
    # Apply mean solution
    es_mean = es.get_mean(state=state)
    ws.update_base_theta(centroids=es_mean)
    ws.set_theta(ws.base_theta)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    
    # Initialize wandb
    if not args.disable_wandb:
        config = vars(args)
        config.update({
            'device': device,
            'total_parameters': None,  # Will be updated later
            'trainer_type': 'hybrid'
        })
        
        run_name = args.wandb_name if args.wandb_name else f"{args.net}_{args.dataset}_hybrid_pop{args.popsize}_ep{args.epochs}"
        init_wandb(
            project_name=args.wandb_project, 
            run_name=run_name,
            config=config,
            tags=[args.net, args.dataset, "hybrid"]
        )
    
    # Create save directory
    save_path = args.save_path + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    print(f"Created save directory: {save_path}")
    
    # Setup logging
    log_file = os.path.join(save_path, 'training.log')
    logger = Logger(log_file)
    sys.stdout = logger
    
    # Load data and model
    train_loader, val_loader, test_loader, num_classes = load_data(args.dataset, args.batch_size)
    model = get_model(args.net, num_classes, device)
    
    # Setup loss function
    criterion = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.CrossEntropyLoss() if args.loss_fn == 'ce' else f1_score_error
    
    # Load or train initial model
    checkpoint_path = f"logs/checkpoints/{args.net}_{args.dataset}_checkpoint_0.pth.tar"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    else:
        sgd_finetune(model, train_loader, criterion, steps=args.steps_finetune, lr=args.lr, device=device)
        os.makedirs("logs/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        
    # freeze_bn(model)
    
    # Setup decoder
    base_theta = flatten_params(model.parameters())
    train_iter = itertools.cycle(train_loader)
    batch = next(train_iter)
    num_weights = len(base_theta)
    num_weight_groups = args.k
    num_decision_vars = num_weight_groups
    
    print(f"Total trainable weights: {num_weights}")
    print(f"Number of weight groups: {num_weight_groups}")
    print(f"Number of decision variables: {num_decision_vars}")
    
    # Update wandb config with model info
    import wandb
    if wandb.run is not None:
        wandb.config.update({
            'total_parameters': num_weights, 
            'num_weight_groups': num_weight_groups,
            'num_decision_vars': num_decision_vars
        })
    
    
    ws = GMMSoftWeightSharing(model=model, criterion=criterion, initial_theta=base_theta, K=num_weight_groups, 
                              sigma_k2=1.0, sigma_eff2=0.01, lambda_reg=0.01,
                              subsample_frac=0.01, weight_batch_size=2**20, device=device)
    num_dims = ws.K
    # Initialize optimization
    f_rand = ws.objective(centroids=np.random.randn(num_dims) * 0.001, batch=batch)
    print(f"fitness of random weights: {f_rand}")
    
    # Evaluate initial model
    acc = evaluate_model_acc(model=model, data_loader=test_loader, device=device, train=False)
    test_acc_tracker = {0: acc}
    print(f"Test accuracy of random weights: {acc:.4f}")
    
    # Setup optimization
    key = jax.random.PRNGKey(args.seed)

    es, es_params = strategy_init(strategy=args.strategy, args=args, x0=np.zeros(num_dims), steps=args.steps_finetune)
    state = es.init(key=key, mean=np.zeros(num_dims), params=es_params)
    
    # Evaluate zero solution
    f_0 = ws.objective(centroids=np.zeros(num_dims), batch=batch)
    acc_0 = evaluate_model_acc(model=model, data_loader=test_loader, device=device, train=False)
    print(f"Test accuracy of zero weights: {acc_0:.4f}, fitness: {f_0:.6f}")
    
    # Main optimization loop
    countfe, epoch = 0, 1
    while countfe < args.max_fe:
        # Gradient-free optimization
        gradfree_trainer(es, es_params, state, key, ws, train_loader, epoch)

        acc = evaluate_model_acc(model=model, data_loader=test_loader, device=device, train=False)
        test_acc_tracker[epoch] = acc
        plot_convergence(acc_tracker=test_acc_tracker, save_path=save_path)
        print(f"Epoch {epoch} after gradient-free optimization, Test acc: {acc:.4f}")
        
        # Log gradient-free phase metrics to wandb
        log_training_metrics(
            epoch=epoch,
            batch_idx=0,
            train_fitness=0.0,  # Not available in this context
            test_loss=0.0,  # Not computed here
            test_top1=acc,
            additional_metrics={'phase': 'gradient_free', 'function_evaluations': countfe}
        )

        torch.save({
                    'state_dict': model.state_dict(),
                    'test_acc_tracker': test_acc_tracker,
                    'args': args,
            }, f"{save_path}/checkpoint_{epoch}.pth.tar")
        print(f"Saved checkpoint to {save_path}/checkpoint_{epoch}.pth.tar")
        epoch += 1

        # Gradient-based optimization
        sgd_finetune(model, train_loader, criterion, steps=args.steps_finetune, lr=args.lr, device=device)
        
        acc = evaluate_model_acc(model=model, data_loader=test_loader, device=device, train=False)
        test_acc_tracker[epoch] = acc
        plot_convergence(acc_tracker=test_acc_tracker, save_path=save_path)
        print(f"Epoch {epoch} after gradient-based optimization, Test acc: {acc:.4f}")
        
        # Log gradient-based phase metrics to wandb
        log_training_metrics(
            epoch=epoch,
            batch_idx=0,
            train_fitness=0.0,  # Not available in this context
            test_loss=0.0,  # Not computed here
            test_top1=acc,
            additional_metrics={'phase': 'gradient_based', 'function_evaluations': countfe}
        )

        torch.save({
                    'state_dict': model.state_dict(),
                    'test_acc_tracker': test_acc_tracker,
                    'args': args,
            }, f"{save_path}/checkpoint_{epoch}.pth.tar")
        print(f"Saved checkpoint to {save_path}/checkpoint_{epoch}.pth.tar")
        epoch += 1
    
    # Finish wandb run
    finish_wandb_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_fe', type=int, default=300000)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--popsize', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_finetune', type=float, default=1e-2)
    parser.add_argument('--steps_finetune', type=int, default=190)
    parser.add_argument('--save_path', type=str, default='logs/results')
    parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'f1'])
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--decoder_type', type=str, default='weight_sharing', choices=['weight_sharing', 'gradient_sharing', 'difference'])
    parser.add_argument('--probability', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--strategy', type=str, default='Open_ES', choices=['CMA_ES', 'SimpleES', 'Open_ES', 'SV_Open_ES'])
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='evogradnn', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (auto-generated if None)')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    print(f"Starting training with arguments: {args}")
    main(args)
