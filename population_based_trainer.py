import argparse
import itertools
import os
import sys
import time

import jax
from matplotlib import pyplot as plt
import numpy as np
import torch
from weight_sharing import *
from models import get_model
from utils import Logger, evaluate_model, evaluate_model_acc, f1_score_error, load_data, params_to_vector, set_seed, population_based_strategy_init, init_wandb, log_training_metrics, log_evolution_metrics, finish_wandb_run
import wandb



def evaluate_model_on_batch(model, criterion, batch, device):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        inputs, targets = batch[0].to(device), batch[1].to(device)
        output = model(inputs)
        loss = criterion(output, targets)
    return loss.item()


def evaluate_solution_on_batch(solution, ws, batch, device):
    x = ws.expand(solution)
    theta = ws.process(x)
    theta = theta.to(device)
    ws.load_to_model(theta)
    fitness = evaluate_model_on_batch(model=ws.model, criterion=ws.criterion, batch=batch, device=device)
    return fitness


def evaluate_population_on_batch(population, ws, batch, device):
    fitnesses = np.zeros(len(population))
    for i, z in enumerate(population):
        x = ws.expand(z)
        theta = ws.process(x)
        theta = theta.to(device)
        ws.load_to_model(theta)
        fitnesses[i] = evaluate_model_on_batch(model=ws.model, criterion=ws.criterion, batch=batch, device=device)
    return fitnesses


def gradfree_trainer(es, es_params, es_state, key, ws, train_loader, test_loader, epoch, save_path, device):
    
    z0 = es.get_population(state=es_state).mean(axis=0)
    train_loader_iterator = itertools.cycle(train_loader)
    num_batches = len(train_loader)
    for count_batch in range(1, num_batches+1):
        batch = next(train_loader_iterator)
        population = es.get_population(state=es_state)
        fitnesses = evaluate_population_on_batch(population=population, ws=ws, batch=batch, device=device)
        es_state = es_state.replace(fitness=fitnesses)

        for count_iter in range(1, 11):    
            # Get new solutions
            key, key_ask, key_tell = jax.random.split(key, 3)
            solutions, es_state = es.ask(key_ask, es_state, es_params)
            
            # Evaluate solutions
            fitnesses = evaluate_population_on_batch(population=solutions, ws=ws, batch=batch, device=device)
                    
            # Update ES
            es_state, metrics = es.tell(key=key_tell, population=solutions, fitness=fitnesses, state=es_state, params=es_params)

            # Apply mean solution
            es_mean_z = es.get_population(state=es_state).mean(axis=0)
            es_mean_fitness = evaluate_solution_on_batch(solution=es_mean_z, ws=ws, batch=batch, device=device)

            # if count_iter % 100 == 0:
            #     model_params = params_to_vector(ws.model.parameters(), to_numpy=True)
                # plot_param_distribution(model_params, save_path=os.path.join(save_path, "param_distribution", f"epoch{epoch}_iter{count_iter}.pdf"))

            # ws.update_base(theta=flatten_params(ws.model.parameters()))
            # state = state.replace(mean=state.mean * 0)

            # Log metrics
            print(f"Epoch {epoch}, batch {count_batch}, iteration {count_iter}, "
                    f"best fitness: {metrics['best_fitness_in_generation']:.6f}, "
                    f"avg pop fitness: {np.mean(fitnesses):.6e}, "
                    f"mean fitness: {es_mean_fitness:.6f}, "
                    f"best solution norm: {metrics['best_solution_norm']:.6f}")
    
        mean_test_loss, mean_test_top1 = evaluate_model(model=ws.model, criterion=ws.criterion, data_loader=test_loader, device=device, train=False)

        delta = es_mean_z - z0
        delta_norm = np.linalg.norm(delta)
        z0 = z0 + 0.1 * delta
        population = np.random.normal(z0, 0.1, size=population.shape)
        es_state = es_state.replace(population=population)

        # Log to wandb
        log_evolution_metrics(
            epoch=epoch,
            iteration=count_iter,
            best_fitness=metrics['best_fitness_in_generation'],
            avg_pop_fitness=np.mean(fitnesses),
            best_solution_norm=metrics['best_solution_norm'],
            mean_fitness=es_mean_fitness,
            test_top1=mean_test_top1,
            test_loss=mean_test_loss,
            additional_metrics={'batch_idx': count_batch, 
                                'Evolution/delta_norm': delta_norm}
        )

    return es_state


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    
    # Initialize wandb
    if not args.disable_wandb:
        config = vars(args)
        config.update({
            'device': device,
            'num_weights': 0,  # Will be updated later
            'trainer_type': 'population_based'
        })
        
        run_name = args.wandb_name if args.wandb_name else f"{args.model}_{args.dataset}_{args.strategy}_pop{args.popsize}_ep{args.epochs}"
        init_wandb(
            project_name=args.wandb_project, 
            run_name=run_name,
            config=config,
            tags=[args.model, args.dataset, args.strategy, "population_based"]
        )
    
    # Create save directory
    save_path = args.save_path + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    print(f"Created save directory: {save_path}")
    os.makedirs(os.path.join(save_path, "param_trajectory"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "param_distribution"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(save_path, 'training.log')
    logger = Logger(log_file)
    sys.stdout = logger
    
    # Load data and model
    train_loader, val_loader, test_loader, num_classes = load_data(args.dataset, args.batch_size)
    model = get_model(args.model, num_classes, device)
    
    # Setup loss function
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss() if args.loss_fn == 'ce' else f1_score_error
    go_criterion = torch.nn.CrossEntropyLoss()
    
    # Load or train initial model
    # checkpoint_path = f"logs/checkpoints/{args.net}_{args.dataset}_checkpoint_0.pth.tar"
    # if os.path.exists(checkpoint_path):
    #     model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    # else:
    #     sgd_finetune(model, train_loader, go_criterion, steps=args.steps_finetune, lr=args.lr, device=device)
    #     os.makedirs("logs/checkpoints", exist_ok=True)
    #     torch.save(model.state_dict(), checkpoint_path)
    
    # Setup decoder
    base_theta = params_to_vector(model.parameters(), to_numpy=True)
    train_iter = itertools.cycle(train_loader)
    batch = next(train_iter)
    test_batch = next(iter(test_loader))
    num_weights = len(base_theta)
    num_K = 128
    num_Z = 128
    print(f"Total trainable weights: {num_weights}")
    
    # Update wandb config with model info
    if wandb.run is not None:
        wandb.config.update({'num_weights': num_weights, 'num_K': num_K, 'num_Z': num_Z}, allow_val_change=True)
    
    param_sharing_type = 'WeightSoftSharingWithDecoder'

    if param_sharing_type == 'hard':
        ws = ParameterSharing(model=model, criterion=criterion, K=num_K, device=device, seed=args.seed)
    elif param_sharing_type == 'GaussianRBF':
        ws = GaussianRBFParameterSharing(model=model, criterion=criterion, K=num_K, device=device, seed=args.seed)
    elif param_sharing_type == 'BlockwiseFixedWeightSoftSharing':
        ws = BlockwiseFixedWeightSoftSharing(model=model, criterion=criterion, K=num_K, device=device, seed=args.seed)
    elif param_sharing_type == 'BlockwiseSoftAttentionSharing':
        ws = BlockwiseSoftAttentionSoftSharing(model=model, criterion=criterion, K=num_K, Z=num_Z, device=device, seed=args.seed)
    elif param_sharing_type == 'WeightSoftSharingWithDecoder':
        ws = WeightSoftSharingWithDecoder(model=model, criterion=criterion, K=num_K, Z=num_Z, device='cpu', seed=args.seed)
    num_dims = ws.num_dims
    
    # Initialize optimization
    x0 = ws.x0
    
    # Evaluate initial model
    acc = evaluate_model_acc(model=model, data_loader=test_loader, device=device, train=False)
    test_acc_tracker = {0: acc}
    print(f"Test accuracy of random weights: {acc:.4f}")
    
    # Setup optimization
    key = jax.random.PRNGKey(args.seed)

    es, es_params = population_based_strategy_init(strategy=args.strategy, args=args, x0=x0, steps=len(train_loader))
    init_population = np.random.uniform(-1, 1, size=(args.popsize, num_dims))
    init_fitness = np.zeros(args.popsize)
    init_fitness = evaluate_population_on_batch(population=init_population, ws=ws, batch=batch, device=device)
    es_state = es.init(key=key, 
                       population=init_population,
                       fitness=init_fitness,
                       params=es_params)
    # Main optimization loop
    countfe, epoch = 0, 1
    while epoch <=  args.epochs:
        es_state = gradfree_trainer(es, es_params, es_state, key, ws, train_loader, test_loader, epoch, save_path, device)
        # Evaluate periodically
        if True:
            # es_mean_z = es.get_population(state=es_state).mean(axis=0)
            # es_mean_fitness = evaluate_solution_on_batch(solution=es_mean_z, ws=ws, batch=batch, device=device)
            es_mean_train_loss, es_mean_train_top1 = evaluate_model(model=ws.model, criterion=criterion, data_loader=train_loader, device=device, train=False)
            es_mean_test_loss, es_mean_test_top1 = evaluate_model(model=ws.model, criterion=criterion, data_loader=test_loader, device=device, train=False)
            
            # Log main training metrics to wandb
            log_training_metrics(
                epoch=epoch,
                batch_idx=0,  # End of epoch evaluation
                train_loss=es_mean_train_loss,
                train_top1=es_mean_train_top1,
                test_loss=es_mean_test_loss,
                test_top1=es_mean_test_top1
            )
            
        # Save checkpoint periodically
        if epoch % 20 == 0:
            torch.save({
                'state_dict': model.state_dict()
            }, os.path.join(save_path, "checkpoints", f"checkpoint_{epoch}.pth.tar"))
            print(f"Saved checkpoint to {os.path.join(save_path, 'checkpoints', f'checkpoint_{epoch}.pth.tar')}")

        epoch += 1
    
    # Finish wandb run
    finish_wandb_run()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--popsize', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_finetune', type=float, default=1e-2)
    parser.add_argument('--steps_finetune', type=int, default=190)
    parser.add_argument('--save_path', type=str, default='logs/results')
    parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'f1'])
    parser.add_argument('--strategy', type=str, default='PSO', choices=['DE', 'PSO', 'DiffusionEvolution'])
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='evo-training', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (auto-generated if None)')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    print(f"Starting training with arguments: {args}")
    main(args)
