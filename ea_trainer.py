import argparse
import itertools
import os
import sys
import copy
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


def evaluate_solution_on_batch(solution, ws, batch, weight_decay=0, device='cuda'):
    load_solution_to_model(solution, ws, device)
    fitness = evaluate_model_on_batch(model=ws.model, criterion=ws.criterion, batch=batch, device=device)
    theta = params_to_vector(ws.model.parameters(), to_numpy=True)
    theta_norm = np.linalg.norm(theta)
    fitness = fitness + weight_decay * theta_norm
    return fitness


def evaluate_population_on_batch(population, ws, batch, weight_decay=0, device='cuda'):
    fitnesses = np.zeros(len(population))
    for i, z in enumerate(population):
        load_solution_to_model(z, ws, device)
        fitnesses[i] = evaluate_model_on_batch(model=ws.model, criterion=ws.criterion, batch=batch, device=device)
        theta = params_to_vector(ws.model.parameters(), to_numpy=True)
        theta_norm = np.linalg.norm(theta)
        fitnesses[i] = fitnesses[i] + weight_decay * theta_norm
    return fitnesses


def population_trainer(es, es_params, es_state, key, ws, train_loader, test_loader, epoch, num_iterations, learning_rate, device, args):

    weight_decay = args.wd
    momentum = args.momentum

    train_loader_iterator = itertools.cycle(train_loader)
    num_batches = len(train_loader)
    period = num_batches // 10
    
    # Hybrid approach: Population evolution for local exploration + momentum for batch accumulation
    # This prevents overfitting population mean to individual random batches
    z0 = es.get_population(state=es_state).mean(axis=0)
    v = np.zeros(len(z0))
    mean_z = copy.deepcopy(z0)

    mean_loss_meter = AverageMeter(name='mean_loss', fmt=':.4e')
    pop_avg_loss_meter = AverageMeter(name='pop_avg_loss', fmt=':.4e')
    best_norm_meter = AverageMeter(name='best_norm', fmt=':.4e')
    mean_norm_meter = AverageMeter(name='mean_norm', fmt=':.4e')

    for count_batch in range(1, num_batches+1):
        batch = next(train_loader_iterator)

        for count_iter in range(1, num_iterations+1):
            # Get new solutions
            key, key_ask, key_tell = jax.random.split(key, 3)
            solutions, es_state = es.ask(key_ask, es_state, es_params)
            
            # Evaluate solutions
            fitnesses = evaluate_population_on_batch(population=solutions, ws=ws, batch=batch, device=device, weight_decay=weight_decay)

            # Update ES
            es_state, metrics = es.tell(key=key_tell, population=solutions, fitness=fitnesses, state=es_state, params=es_params)

            # Apply mean solution
            mean_z = es.get_population(state=es_state).mean(axis=0)
            mean_loss_meter.update(evaluate_solution_on_batch(solution=mean_z, ws=ws, batch=batch, device=device))

            best_z_norm = np.linalg.norm(solutions[np.argmin(fitnesses)])
            mean_z_norm = np.linalg.norm(mean_z)
            pop_avg_loss_meter.update(np.mean(fitnesses))
            best_norm_meter.update(best_z_norm)
            mean_norm_meter.update(mean_z_norm)
        
        # Extract signal from population evolution, apply momentum accumulation
        current_population_mean = es.get_population(state=es_state).mean(axis=0)
        delta = current_population_mean - z0  # Direction learned from population on this batch
        delta_norm = np.linalg.norm(delta)
        v = momentum * v + learning_rate * delta
        z0 = z0 + v
        z0_norm = np.linalg.norm(z0)
        
        load_solution_to_model(z0, ws, device)

        # Log to wandb
        if (count_batch - 1) % period == 0:
            theta = params_to_vector(ws.model.parameters(), to_numpy=True)
            theta_norm = np.linalg.norm(theta)
            
            log_evolution_metrics(
                epoch=epoch,
                batch=count_batch,
                metrics={
                    'Evolution/pop_best_loss': metrics['best_fitness_in_generation'],
                    'Evolution/pop_avg_loss': pop_avg_loss_meter.avg,
                    'Evolution/mean_loss': mean_loss_meter.avg,
                    'Evolution/best_solution_norm': best_norm_meter.avg,
                    'Evolution/delta_norm': delta_norm,
                    'Evolution/population_mean_norm': np.linalg.norm(current_population_mean),
                    'Evolution/momentum_solution_norm': z0_norm,
                    'Evolution/theta_norm': theta_norm
                }
            )
            print(f"Epoch {epoch}, batch {count_batch}, "
                    f"best pop loss: {metrics['best_fitness_in_generation']:.4f}, "
                    f"avg pop loss: {pop_avg_loss_meter.avg:.4f}, "
                    f"mean loss: {mean_loss_meter.avg:.4f}")

        # Reset population around momentum-updated position for next batch
        # This prevents population from overfitting to this specific batch
        new_population = np.random.normal(z0, args.std, size=(args.popsize, len(z0)))
        es_state = es_state.replace(population=new_population)
    
    return key, es_state


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
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
    batch_size = args.batch_size
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    model = get_model(args.arch, num_classes, device)
    base_theta = params_to_vector(model.parameters(), to_numpy=True)
    num_weights = len(base_theta)
    
    # Setup loss function
    criterion = torch.nn.CrossEntropyLoss() if args.criterion.lower() == 'ce' else f1_score_error
    
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
        hidden_dims = args.hidden_dims if args.hidden_dims is not None else [128, 32]
        ws = MLPSoftSharing(model=model, criterion=criterion, d=d, hidden_dims=hidden_dims, device=args.ws_device, seed=args.seed)
    elif param_sharing_type == 'hypernetworksoftsharing':
        ws = HyperNetworkSoftSharing(model=model, criterion=criterion, d=d, device=args.ws_device, seed=args.seed)
    
    # Setup optimization
    key = jax.random.PRNGKey(0)

    x0 = np.zeros(d)

    es, es_params = population_based_strategy_init(strategy=args.optimizer, args=args, x0=x0, steps=len(train_loader))
    # Initialize population state
    init_population = np.random.normal(x0, args.std, size=(args.popsize, d))
    init_fitness = np.zeros(args.popsize)
    batch = next(iter(train_loader))
    init_fitness = evaluate_population_on_batch(population=init_population, ws=ws, batch=batch, device=device)
    es_state = es.init(key=key, 
                       population=init_population,
                       fitness=init_fitness,
                       params=es_params)
    
    num_iterations = args.num_iterations

    scheduler = CosineAnnealingScheduler(eta_max=args.lr, eta_min=1e-4, T_max=args.epochs, T_mult=1)

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
            'hidden_dims': args.hidden_dims,
        }
        run_name = args.wandb_name if args.wandb_name else f"{timestamp}"
        init_wandb(
            project_name=f"{args.wandb_project}-{args.dataset}-{args.arch}", 
            run_name=run_name,
            config=config,
            tags=[args.arch, args.dataset, args.optimizer, args.ws_type, "population_based"]
        )
    # Main optimization loop
    best_acc= 0.0
    epoch = 1
    while epoch <=  args.epochs:
        lr = scheduler.get_lr()

        key, es_state = population_trainer(es, es_params, es_state, key, 
                                    ws, train_loader, test_loader, 
                                    epoch, num_iterations, lr, 
                                    device, args)

        es_mean_test_ce, es_mean_test_top1 = evaluate_model(model=ws.model, 
                                                            criterion=ws.criterion, 
                                                            data_loader=test_loader, 
                                                            device=device, 
                                                            train=False)
            
        # Log main training metrics to wandb
        log_evaluation_metrics(
            epoch=epoch,
            metrics={'lr': lr,
                     'Test/loss': es_mean_test_ce,
                     'Test/top1': es_mean_test_top1}
        )

        print(f"Epoch {epoch}, mean test loss: {es_mean_test_ce:.3f}, mean test top1: {es_mean_test_top1:.3f}")
            
        if best_acc < es_mean_test_top1:
            best_acc = es_mean_test_top1
            print(f"New best top1: {best_acc:.3f}")
            torch.save({
                'state_dict': model.state_dict()
            }, os.path.join(save_path, "checkpoints", f"best_checkpoint.pth.tar"))
            print(f"Saved checkpoint to {os.path.join(save_path, 'checkpoints', f'best_checkpoint.pth.tar')}")

        epoch += 1

        scheduler.step()

        # Reset base_theta and population state when coordinate system changes
        base_theta = params_to_vector(ws.model.parameters())
        ws.set_theta(base_theta)
        # Reset population state when base_theta changes - the coordinate system has shifted
        init_population = np.random.normal(np.zeros(args.d), args.std, size=(args.popsize, args.d))
        init_fitness = np.zeros(args.popsize)
        batch = next(iter(train_loader))
        init_fitness = evaluate_population_on_batch(population=init_population, ws=ws, batch=batch, device=device)
        es_state = es.init(key=key, 
                           population=init_population,
                           fitness=init_fitness,
                           params=es_params)
    
    torch.save({
                'state_dict': model.state_dict()
            }, os.path.join(save_path, "checkpoints", f"final_checkpoint.pth.tar"))
    print(f"Saved final checkpoint to {os.path.join(save_path, 'checkpoints', f'final_checkpoint.pth.tar')}")
    log_dict = {
        'Final/test_top1': best_acc,
        'Final/test_loss': es_mean_test_ce,
    }
    # Finish wandb run
    finish_wandb_run(log_dict)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet32')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_iterations', type=int, default=8)
    parser.add_argument('--d', type=int, default=1024)
    parser.add_argument('--popsize', type=int, default=30)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--save_path', type=str, default='logs/results')
    parser.add_argument('--ws_type', type=str, default='hypernetworksoftsharing', 
                        choices=['hard', 'gaussianrbf', 'mlpsoftsharing', 'hypernetworksoftsharing', 'perceptronsoftsharing'])
    parser.add_argument('--hidden_dims', type=int, default=None, nargs='+', help='Hidden dimensions for MLP soft sharing')
    parser.add_argument('--criterion', type=str, default='ce', choices=['ce', 'f1'])
    parser.add_argument('--optimizer', type=str, default='PSO', choices=['DE', 'PSO', 'DiffusionEvolution'])
    parser.add_argument('--ws_device', type=str, default='cuda', choices=['cuda', 'cpu'])
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='evo-weight-sharing', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (auto-generated if None)')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')

    args = parser.parse_args()
    main(args)
