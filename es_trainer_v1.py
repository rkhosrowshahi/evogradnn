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


def train_epoch(es, es_params, es_state, key, ws, train_loader, val_loader, epoch, num_iterations, learning_rate, device, args):

    weight_decay = args.wd
    momentum = args.momentum

    train_loader_iterator = itertools.cycle(train_loader)
    num_batches = len(train_loader)
    period = num_batches // 10
    
    # Hybrid approach: ES for local exploration + momentum for batch accumulation
    # This prevents overfitting ES mean to individual random batches
    z0 = es.get_mean(state=es_state)  # Starting point from ES
    v = es.get_mean(state=es_state)   # Momentum velocity

    mean_loss_meter = AverageMeter(name='mean_loss', fmt=':.4e')
    pop_best_loss_meter = AverageMeter(name='pop_best_loss', fmt=':.4e')
    pop_avg_loss_meter = AverageMeter(name='pop_avg_loss', fmt=':.4e')
    best_norm = 0
    mean_norm = 0

    for count_batch in range(1, num_batches+1):
        batch = next(train_loader_iterator)
        
        z0 = es.get_mean(state=es_state)

        for count_iter in range(1, num_iterations+1):
            # Get new solutions
            key, key_ask, key_tell = jax.random.split(key, 3)
            solutions, es_state = es.ask(key_ask, es_state, es_params)
            
            # Evaluate solutions
            fitnesses = evaluate_population_on_batch(population=solutions, ws=ws, batch=batch, device=device, weight_decay=weight_decay)

            # Update ES
            es_state, metrics = es.tell(key=key_tell, population=solutions, fitness=fitnesses, state=es_state, params=es_params)

            # Apply mean solution
            zt = es.get_mean(state=es_state)
            mean_loss_meter.update(evaluate_solution_on_batch(solution=zt, ws=ws, batch=batch, device=device))

            best_norm = np.linalg.norm(solutions[np.argmin(fitnesses)])
            mean_norm = np.linalg.norm(zt)
            pop_best_loss_meter.update(np.min(fitnesses))
            pop_avg_loss_meter.update(np.mean(fitnesses))
        
        # Extract signal from ES exploration, apply momentum accumulation
        zt = es.get_mean(state=es_state)
        
        delta = zt - z0  # Direction learned from ES on this batch
        delta_norm = np.linalg.norm(delta)
        # delta *= min(1.0, 2.0 / (delta_norm + 1e-8))

        lr_t = learning_rate
        
        # Momentum update to accumulate learning across batches
        v = momentum * v + lr_t * delta

        # Log to wandb
        if count_batch == 1 or (count_batch - 1) % period == 0 or count_batch == num_batches:
            # Load momentum-accumulated solution
            load_solution_to_model(z0, ws, device)
            theta_z0 = params_to_vector(ws.model.parameters(), to_numpy=True)
            theta_z0_norm = np.linalg.norm(theta_z0)

            theta_z0_test_loss, theta_z0_test_top1 = evaluate_model(model=ws.model, criterion=ws.criterion, 
                                                                data_loader=val_loader, 
                                                                device=device, 
                                                                train=False)

            load_solution_to_model(zt, ws, device)
            theta_zt = params_to_vector(ws.model.parameters(), to_numpy=True)
            theta_zt_norm = np.linalg.norm(theta_zt)
            theta_zt_test_loss, theta_zt_test_top1 = evaluate_model(model=ws.model, criterion=ws.criterion, 
                                                                data_loader=val_loader, 
                                                                device=device, 
                                                                train=False)
            
            
            log_evolution_metrics(
                epoch=epoch,
                batch=count_batch,
                metrics={
                    'Evolution/pop_best_loss': pop_best_loss_meter.avg,
                    'Evolution/pop_avg_loss': pop_avg_loss_meter.avg,
                    'Evolution/mean_loss': mean_loss_meter.avg,
                    'Evolution/best_norm': best_norm,
                    'Evolution/mean_norm': mean_norm,
                    'Evolution/momentum_norm': np.linalg.norm(v),
                    'Evolution/delta_norm': delta_norm,
                    'Evolution/sigma': np.mean(es_state.std),
                    'Evolution/theta_z0_norm': theta_z0_norm,
                    'Evolution/theta_z0_test_top1': theta_z0_test_top1,
                    'Evolution/theta_z0_test_loss': theta_z0_test_loss,
                    'Evolution/theta_zt_norm': theta_zt_norm,
                    'Evolution/theta_zt_test_top1': theta_zt_test_top1,
                    'Evolution/theta_zt_test_loss': theta_zt_test_loss,
                    'Evolution/lr': lr_t
                }
            )
            print(f"Epoch {epoch}, batch {count_batch}, "
                    f"best pop loss: {pop_best_loss_meter.avg:.4f}, "
                    f"avg pop loss: {pop_avg_loss_meter.avg:.4f}, "
                    f"mean loss: {mean_loss_meter.avg:.4f}")
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
    train_dataset, val_dataset, test_dataset, num_classes, input_size = get_dataset(args.dataset, validation_split=0.01)
    # train_dataset = create_balanced_dataset(train_dataset, num_classes=num_classes, samples_per_class=103)
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
    model = get_model(model_name=args.arch, input_size=input_size, num_classes=num_classes, device=device)
    theta_0 = params_to_vector(model.parameters(), to_numpy=True)
    num_weights = len(theta_0)
    
    # Setup loss function
    criterion = torch.nn.CrossEntropyLoss() if args.criterion == 'ce' else f1_score_error
    
    # Setup decoder
    d = args.d
    
    param_sharing_type = args.ws_type.lower()

    if param_sharing_type == 'hard':
        ws = ParameterSharing(model=model, criterion=criterion, d=d, device=args.ws_device, seed=args.seed)
    elif param_sharing_type == 'gaussianrbf':
        ws = GaussianRBFParameterSharing(model=model, criterion=criterion, d=d, device=args.ws_device, seed=args.seed)
    elif param_sharing_type == 'randproj':
        ws = RandomProjectionSoftSharing(model=model, criterion=criterion, d=d, device=args.ws_device, seed=args.seed)
    elif param_sharing_type == 'sparseproj':
        ws = SparseRandomProjectionSoftSharing(model=model, criterion=criterion, d=d, device=args.ws_device, seed=args.seed)
    elif param_sharing_type == 'mlp':
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
        ws = MLPSoftSharing(model=model, criterion=criterion, d=d, hidden_dims=hidden_dims, use_activation=args.use_activation, activation=args.activation, device=args.ws_device, seed=args.seed)
    elif param_sharing_type == 'hypernetwork':
        ws = HyperNetworkSoftSharing(model=model, criterion=criterion, d=d, device=args.ws_device, seed=args.seed)
    
    # Setup optimization
    key = jax.random.PRNGKey(0)

    x0 = np.zeros(d)

    es, es_params, es_state = distribution_based_strategy_init(key=key, strategy=args.optimizer, x0=x0, steps=args.num_iterations, args=args)
    
    num_iterations = args.num_iterations

    scheduler = CosineAnnealingScheduler(eta_max=args.lr, eta_min=1e-4, T_max=args.epochs, T_mult=1)

    # Initialize wandb
    if not args.disable_wandb:
        config = {
            'timestamp': timestamp,
            'arch': args.arch,
            'dataset': args.dataset,
            'optimizer': args.optimizer,
            'ws_type': args.ws_type,
            'criterion': args.criterion,
            'seed': args.seed,
            'epochs': args.epochs,
            'num_iterations': args.num_iterations,
            'num_dims': args.d,
            'popsize': args.popsize,
            'std': args.std,
            'lr': args.lr,
            'wd': args.wd,
            'momentum': args.momentum,
            'batch_size': args.batch_size,
            'num_params': num_weights,
            'hidden_dims': args.hidden_dims,
            'use_activation': args.use_activation,
            'activation': args.activation,
            'notes': args.note,
        }
        run_name = args.wandb_name if args.wandb_name else f"{timestamp}"
        init_wandb(
            project_name=f"{args.wandb_project}-{args.dataset}-{args.arch}-v1", 
            run_name=run_name,
            config=config,
            tags=[args.arch, args.dataset, args.optimizer, args.ws_type]
        )
    # Main optimization loop
    best_acc= 0.0
    epoch = 1
    while epoch <=  args.epochs:
        lr = scheduler.get_lr()

        key, es_state = train_epoch(es, es_params, es_state, key, 
                                    ws, train_loader, val_loader, 
                                    epoch, num_iterations, lr, 
                                    device, args)

        # es_mean_train_ce, es_mean_train_top1 = evaluate_model(model=ws.model, criterion=torch.nn.functional.cross_entropy, data_loader=train_loader, device=device, train=False)
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
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--popsize', type=int, default=30)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--save_path', type=str, default='logs/results')
    parser.add_argument('--ws_type', type=str, default='randproj')
    parser.add_argument('--use_activation', action='store_true', help='Use activation function in MLP soft sharing')
    parser.add_argument('--activation', type=str, default=None, choices=['relu', 'tanh', 'gelu', 'leaky_relu'])
    parser.add_argument('--hidden_dims', type=str, default=None, help='Hidden dimensions for MLP soft sharing')
    parser.add_argument('--criterion', type=str, default='ce', choices=['ce', 'f1'])
    parser.add_argument('--optimizer', type=str, default='CMA_ES', choices=['CMA_ES', 'SV_CMA_ES', 'SimpleES', 'Open_ES', 'SV_Open_ES', 'xNES'])
    parser.add_argument('--ws_device', type=str, default='cuda', choices=['cuda', 'cpu'])
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='evo-weight-sharing', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (auto-generated if None)')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--note', type=str, default=None, help='Note for the run')

    args = parser.parse_args()
    main(args)
