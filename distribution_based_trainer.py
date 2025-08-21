import argparse
import itertools
import os
import sys
from datetime import datetime
import jax
import numpy as np
import torch
from torch.utils.data import DataLoader
from weight_sharing import *
from models import get_model
from utils import *
import wandb


def evaluate_model_on_batch(model, criterion, batch, device):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        inputs, targets = batch[0].to(device), batch[1].to(device)
        output = model(inputs)
        loss = criterion(output, targets)
    return loss.item()

def load_solution_to_model(solution, ws, device):
    x = ws.expand(solution)
    theta = ws.process(x)
    theta = theta.to(device)
    ws.load_to_model(theta)


def evaluate_solution_on_batch(solution, ws, batch, device):
    load_solution_to_model(solution, ws, device)
    fitness = evaluate_model_on_batch(model=ws.model, criterion=ws.criterion, batch=batch, device=device)
    return fitness


def evaluate_population_on_batch(population, ws, batch, device):
    fitnesses = np.zeros(len(population))
    for i, z in enumerate(population):
        load_solution_to_model(z, ws, device)
        fitnesses[i] = evaluate_model_on_batch(model=ws.model, criterion=ws.criterion, batch=batch, device=device)
    return fitnesses


def es_trainer(es, es_params, es_state, key, ws, train_loader, test_loader, epoch, num_iterations, learning_rate, weight_decay, momentum, save_path, device):
    
    train_loader_iterator = itertools.cycle(train_loader)
    num_batches = len(train_loader)
    period = int(num_batches * 0.5)
    z0 = es.get_mean(state=es_state)
    v = np.zeros(len(z0))
    es_mean_z = z0
    es_mean_fitness = 0
    es_z0_fitness = 0
    for count_batch in range(1, num_batches+1):
        batch = next(train_loader_iterator)

        es_mean_fitness_new_batch = evaluate_solution_on_batch(solution=es_mean_z, ws=ws, batch=batch, device=device)
        es_z0 = es.get_mean(state=es_state)
        es_z0_fitness_new_batch = evaluate_solution_on_batch(solution=es_z0, ws=ws, batch=batch, device=device)
        
        print(f"Epoch {epoch}, batch {count_batch}, mean fitness: {es_mean_fitness_new_batch:.3f}, z0 fitness: {es_z0_fitness_new_batch:.3f}")

        for count_iter in range(1, num_iterations+1):
            # Get new solutions
            key, key_ask, key_tell = jax.random.split(key, 3)
            solutions, es_state = es.ask(key_ask, es_state, es_params)
            
            # Evaluate solutions
            fitnesses = evaluate_population_on_batch(population=solutions, ws=ws, batch=batch, device=device)

            # Apply weight decay
            norm_solutions = np.linalg.norm(solutions, axis=1)
            fitnesses = fitnesses + weight_decay * norm_solutions
                    
            # Update ES
            es_state, metrics = es.tell(key=key_tell, population=solutions, fitness=fitnesses, state=es_state, params=es_params)
            # print(f"{count_iter} es_state.std: {es_state.std}")

            # Apply mean solution
            es_mean_z = es.get_mean(state=es_state)
            es_mean_fitness = evaluate_solution_on_batch(solution=es_mean_z, ws=ws, batch=batch, device=device)

            best_z_norm = np.linalg.norm(solutions[np.argmin(fitnesses)])
            mean_z_norm = np.linalg.norm(es_mean_z)
        
        delta = es_mean_z - z0
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 1:
            delta = delta / delta_norm
        v = momentum * v + learning_rate * delta
        z0 = z0 + v
        z0_norm = np.linalg.norm(z0)
        
        load_solution_to_model(z0, ws, device)

        # Log to wandb
        if (count_batch - 1) % period == 0:
            mean_test_loss, mean_test_top1 = evaluate_model(model=ws.model, criterion=ws.criterion, data_loader=test_loader, device=device, train=False)
            params = params_to_vector(ws.model.parameters(), to_numpy=True)
            params_norm = np.linalg.norm(params)
            
            log_evolution_metrics(
                epoch=epoch,
                iteration=count_iter,
                best_fitness=metrics['best_fitness_in_generation'],
                avg_pop_fitness=np.mean(fitnesses),
                best_solution_norm=best_z_norm,
                mean_fitness=es_mean_fitness,
                test_top1=mean_test_top1,
                test_loss=mean_test_loss,
                additional_metrics={'batch_idx': count_batch, 
                                    'Evolution/delta_norm': delta_norm,
                                    'Evolution/sigma': np.mean(es_state.std),
                                    'Evolution/mean_solution_norm': mean_z_norm,
                                    'Evolution/base_solution_norm': z0_norm,
                                    'Evolution/params_norm': params_norm}
            )

        es_z0_fitness = evaluate_solution_on_batch(solution=z0, ws=ws, batch=batch, device=device)

        print(f"Epoch {epoch}, batch {count_batch}, mean fitness: {es_mean_fitness:.3f}, z0 fitness: {es_z0_fitness:.3f}, es_state.std: {np.mean(es_state.std):.3e}")

        # es_state = es_state.replace(mean=z0)
        # es_params = es_params.replace(std_init=es_state.std)
        es_state = es.init(key=key, mean=z0, params=es_params)

    return key, es_state


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create save directory
    save_path = os.path.join(args.save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)
    print(f"Created save directory: {save_path}")
    os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(save_path, 'training.log')
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"Starting training with arguments: {args}")
    
    # Load data and model
    train_dataset, val_dataset, test_dataset, num_classes = get_dataset(args.dataset, validation_split=0.01)
    train_dataset = create_balanced_dataset(train_dataset, num_classes=num_classes, samples_per_class=103)
    batch_size = args.batch_size
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )
    model = get_model(args.arch, num_classes, device)
    init_params = params_to_vector(model.parameters(), to_numpy=True)
    num_weights = len(init_params)
    
    # Setup loss function
    criterion = torch.nn.CrossEntropyLoss() if args.criterion == 'ce' else f1_score_error
    
    # Setup decoder
    d = args.d
    
    param_sharing_type = args.ws_type.lower()

    if param_sharing_type == 'hard':
        ws = ParameterSharing(model=model, criterion=criterion, d=d, device=args.ws_device, seed=args.seed)
    elif param_sharing_type == 'gaussianrbf':
        ws = GaussianRBFParameterSharing(model=model, criterion=criterion, d=d, device=args.ws_device, seed=args.seed)
    elif param_sharing_type == 'perceptronsoftsharing':
        ws = PerceptronSoftSharing(model=model, criterion=criterion, d=d, device=args.ws_device, seed=args.seed)
    elif param_sharing_type == 'mlpsoftsharing':
        ws = MLPSoftSharing(model=model, criterion=criterion, d=d, hidden_dims=[16], device=args.ws_device, seed=args.seed)
    elif param_sharing_type == 'hypernetworksoftsharing':
        ws = HyperNetworkSoftSharing(model=model, criterion=criterion, d=d, device=args.ws_device, seed=args.seed)
    
    # Setup optimization
    key = jax.random.PRNGKey(args.seed)

    x0 = np.zeros(d)

    es, es_params, es_state = distribution_based_strategy_init(key=key, strategy=args.optimizer, x0=x0, steps=args.num_iterations, args=args)
    
    num_iterations = args.num_iterations

    scheduler = CosineAnnealingScheduler(eta_max=args.lr, eta_min=1e-5, T_max=args.epochs, T_mult=1)

    # Initialize wandb
    if not args.disable_wandb:
        config = {
            'arch': args.arch,
            'dataset': args.dataset,
            'optimizer': args.optimizer,
            'ws_type': args.ws_type,
            'criterion': args.criterion,
            'seed': args.seed,
            'epochs': args.epochs,
            'num_iterations': args.num_iterations,
            'd': args.d,
            'popsize': args.popsize,
            'std': args.std,
            'lr': args.lr,
            'wd': args.wd,
            'momentum': args.momentum,
            'batch_size': args.batch_size,
            'D': num_weights,
            'd': d,
        }
        run_name = args.wandb_name if args.wandb_name else f"{args.arch}_{args.dataset}_{args.optimizer}_{timestamp}"
        init_wandb(
            project_name=args.wandb_project, 
            run_name=run_name,
            config=config,
            tags=[args.arch, args.dataset, args.optimizer, args.ws_type]
        )
    # Main optimization loop
    epoch = 1
    while epoch <=  args.epochs:
        lr = scheduler.get_lr()

        key, es_state = es_trainer(es, es_params, es_state, key, ws, train_loader, test_loader, epoch, num_iterations, lr, args.wd, args.momentum, save_path, device)

        es_mean_train_ce, es_mean_train_top1 = evaluate_model(model=ws.model, criterion=torch.nn.functional.cross_entropy, data_loader=train_loader, device=device, train=False)
        es_mean_test_ce, es_mean_test_top1 = evaluate_model(model=ws.model, criterion=torch.nn.functional.cross_entropy, data_loader=test_loader, device=device, train=False)
            
        # Log main training metrics to wandb
        log_training_metrics(
            epoch=epoch,
            train_loss=es_mean_train_ce,
            train_top1=es_mean_train_top1,
            test_loss=es_mean_test_ce,
            test_top1=es_mean_test_top1,
            additional_metrics={'lr': lr}
        )
            
        # Save checkpoint periodically
        if epoch % 20 == 0:
            torch.save({
                'state_dict': model.state_dict()
            }, os.path.join(save_path, "checkpoints", f"checkpoint_{epoch}.pth.tar"))
            print(f"Saved checkpoint to {os.path.join(save_path, 'checkpoints', f'checkpoint_{epoch}.pth.tar')}")

        epoch += 1

        scheduler.step()

        # es_params = es_params.replace(std_init=es_params.std_init * 0.9)
        # es_state = es.init(key=key, mean=es_state.mean, params=es_params)
    
    # Finish wandb run
    finish_wandb_run()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_iterations', type=int, default=8)
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--popsize', type=int, default=30)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--save_path', type=str, default='logs/results')
    parser.add_argument('--ws_type', type=str, default='hypernetworksoftsharing', 
                        choices=['hard', 'gaussianrbf', 'mlpsoftsharing', 'hypernetworksoftsharing', 'perceptronsoftsharing'])
    parser.add_argument('--criterion', type=str, default='ce', choices=['ce', 'f1'])
    parser.add_argument('--optimizer', type=str, default='Open_ES', choices=['CMA_ES', 'SV_CMA_ES', 'SimpleES', 'Open_ES', 'SV_Open_ES', 'xNES'])
    parser.add_argument('--ws_device', type=str, default='cuda', choices=['cuda', 'cpu'])
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='evo-weight-sharing-image-classification', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (auto-generated if None)')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')

    args = parser.parse_args()
    main(args)
