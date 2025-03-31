import argparse
import itertools
import jax
from sklearn.metrics import f1_score
from tqdm import tqdm
from models.resnets import ResNet18, ResNet34, ResNet50
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from evosax.algorithms.distribution_based import CMA_ES, Sep_CMA_ES
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.cluster import KMeans
import wandb

from utils import WarmUpLR, load_data, save_model
from models import get_model

print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated



# Fitness Function: Evaluate ResNet-18 on CIFAR-10
def evaluate_model_acc(model, data_loader, device, train=False):
    """
    Evaluate model accuracy on the entire dataset
    Args:
        model: Neural network model
        data_loader: DataLoader containing the dataset
        device: Device to run evaluation on
        train: If True, only evaluate on one batch
    Returns:
        accuracy: Classification accuracy as percentage
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if train:
                correct *= -1  # Negate for minimization in CMA-ES
                break
    accuracy = 100 * correct / total
    return accuracy

def evaluate_model_ce(model, data_loader, device, train=False):
    """
    Evaluate model using cross entropy loss on the entire dataset
    Args:
        model: Neural network model
        data_loader: DataLoader containing the dataset
        device: Device to run evaluation on
        train: If True, only evaluate on one batch
    Returns:
        loss: Average cross entropy loss
    """
    model.eval()
    loss = 0
    total_batch = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += torch.nn.functional.cross_entropy(outputs, labels).item()
            total_batch += 1
            if train:
                break
    if np.isnan(loss) or np.isinf(loss):
        loss = 9.9e+21  # Handle numerical instability
    return loss

def evaluate_model_acc_single_batch(model, batch, device, train=False):
    """
    Evaluate model accuracy on a single batch
    Args:
        model: Neural network model
        batch: Tuple of (inputs, labels)
        device: Device to run evaluation on
        train: Unused parameter for API compatibility
    Returns:
        accuracy: Classification accuracy as percentage
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy * -1

def evaluate_model_f1score_single_batch(model, batch, device, train=False):
    """
    Evaluate model f1 score on a single batch
    Args:
        model: Neural network model
        batch: Tuple of (inputs, labels)
        device: Device to run evaluation on
        train: Unused parameter for API compatibility
    Returns:
        accuracy: Classification accuracy as percentage
    """
    model.eval()
    with torch.no_grad():
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

    f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
    return f1 * -1

def evaluate_model_ce_single_batch(model, batch, device, train=False):
    """
    Evaluate model using cross entropy loss on a single batch
    Args:
        model: Neural network model
        batch: Tuple of (inputs, labels)
        device: Device to run evaluation on
        train: Unused parameter for API compatibility
    Returns:
        loss: Cross entropy loss for the batch
    """
    model.eval()
    loss = 0
    with torch.no_grad():
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += torch.nn.functional.cross_entropy(outputs, labels).item()

    if np.isnan(loss) or np.isinf(loss):
        loss = 9.9e+21  # Handle numerical instability
    return loss


def compute_l2_norm(x) -> np.ndarray:
    """
    Compute L2-norm of x_i. Assumes x to have shape (popsize, num_dims)
    Args:
        x: Input array
    Returns:
        Mean of squared values
    """
    return np.nanmean(x * x)

def build_model(model, W, total_weights, solution, codebook, cma, weight_offsets, device='cuda'):
    """
    Build model using the solution parameters from CMA-ES
    Args:
        model: Base neural network model
        W: Number of components
        total_weights: Total number of parameters
        solution: Solution vector from CMA-ES
        codebook: Dictionary mapping components to parameter indices
        cma: CMA-ES state
        weight_offsets: Random offsets for parameters
        device: Device to place model on
    Returns:
        model: Updated model with new parameters
    """
    solution = np.array(solution)
    means = solution[:W]
    log_sigmas = solution[W:]
    sigmas = np.exp(log_sigmas)

    # Initialize parameter vector
    params = torch.zeros(total_weights, device=device)
    for k in range(W):
        indices = codebook[k]
        size = len(indices)
        if size > 0: 
            mean_tensor = torch.tensor(means[k], device=device)
            sigma_tensor = torch.tensor(sigmas[k], device=device)
            
            params[indices] = torch.normal(
                mean=mean_tensor,
                std=sigma_tensor,
                size=(size,),
                device=device
            )
    # Assign weights to model
    torch.nn.utils.vector_to_parameters(params, model.parameters())

    return model

def train_on_gd(model, train_loader, optimizer, criterion, step=0, warmup_scheduler=None, device='cuda'):
    """
    Train model using gradient descent
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to run training on
    Returns:
        total_fe: Number of function evaluations
        running_loss: Average loss over all batches
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_epochs = 10
    total_fe = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        total_fe += 1

        if step <= args.warm and warmup_scheduler is not None:
            # print(f"Step {step}, Warmup Scheduler")
            warmup_scheduler.step()
            # print(f"lr: {optimizer.param_groups[0]['lr']}")
    
    running_loss /= total_fe
    return total_fe, running_loss

def ubp_cluster(W, params):
    """
    Uniform bin partitioning clustering
    Args:
        W: Number of bins/clusters
        params: Parameters to cluster
    Returns:
        codebook: Dictionary mapping cluster indices to parameter indices
        centers: Cluster centers
        bin_indices: Cluster assignments for each parameter
    """
    # Calculate bin edges
    min_val = params.min()
    max_val = params.max()
    bins = np.linspace(min_val, max_val, W)
    bin_indices = np.digitize(params, bins) - 1
    
    # Create codebook and compute centers
    centers = []
    log_sigmas = []
    counter = 0
    codebook = {}
    for i in range(W):
        mask = np.where(bin_indices == i)[0]
        if len(mask) == 0:
            continue
        centers.append(params[mask].mean())
        log_sigmas.append(np.log(params[mask].std() + 1e-8))
        bin_indices[mask] = counter
        codebook[counter] = mask

        counter+=1
    centers = np.array(centers)
    log_sigmas = np.array(log_sigmas)
    # Replace NaN values in log_sigmas with log(0.01) as default
    # log_sigmas = np.nan_to_num(log_sigmas, nan=0)
    return codebook, centers, log_sigmas, bin_indices


def random_codebook_initialization(W_init, total_weights):
    weight_indices = np.arange(0, total_weights)
    np.random.shuffle(weight_indices)
    codebook = {}
    d = np.random.dirichlet(np.ones(W_init))
    start_idx = 0
    end_idx = 0
    for key in range(W_init):
        size = np.ceil(d[key] * total_weights).astype(int)
        start_idx = key * size
        end_idx = start_idx + size
        indices = weight_indices[start_idx:end_idx]
        if len(indices) == 0:
            indices = weight_indices[start_idx:end_idx+1]
        codebook[key] = indices
        weight_indices[indices] = np.full((len(indices), ), key)
    return codebook

# CMA-ES Training Loop
def main(args):
    """
    Main training loop using CMA-ES
    Args:
        args: Command line arguments
    """
    # Initialize wandb logging
    wandb.init(project=f"GF-CMA_ES-{args.net}-{args.dataset}", config={
        "W_init": args.w_init,
        "steps": args.steps,
        "popsize": args.popsize,
        "device": args.device,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "net": args.net,
        "sigma_init": args.sigma_init,
        "lr_init": args.lr_init,
        "dataset": args.dataset,
        "weight_decay": args.weight_decay,
        "lr_decay": args.lr_decay,
        "eval_interval": args.eval_interval,
        "gd_interval": args.gd_interval,
        "objective": args.objective
    })

    # Setup evaluation function and device
    obj = evaluate_model_ce
    if args.objective == 'acc':
        obj = evaluate_model_acc_single_batch
    elif args.objective == 'f1':
        obj = evaluate_model_f1score_single_batch  
    elif args.objective == 'ce':
        obj = evaluate_model_ce_single_batch
    # evaluate = evaluate_model_acc
    # Initialize ResNet-18
    device = args.device
    W_init = args.w_init
    popsize = args.popsize
    steps = args.steps
    
    train_loader, val_loader, test_loader, num_classes = load_data(args.dataset, args.batch_size)

    model = get_model(args.net, num_classes, device)

    # print(evaluate(model, train_loader, device, train=True))
    total_weights = sum(p.numel() for p in model.parameters())
    initial_weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()

    codebook = {}
    # codebook = random_codebook_initialization(W_init, total_weights)
         
    weight_offsets = torch.normal(mean=0.0, std=1.0, size=(total_weights,), device=device)

    D = W_init * 2
    # Initialize CMA-ES
    rng = jax.random.PRNGKey(0)
    x0 = np.concatenate([np.zeros(W_init), np.full(W_init, np.log(0.01))])
    solver = CMA_ES(population_size=popsize, solution=x0)
    es_params = solver.default_params
    es_params = es_params.replace(std_init=args.sigma_init)
    state = solver.init(rng, x0, es_params)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    # train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    
    total_fe = 0
    train_iter = itertools.cycle(train_loader)
    state_wandb = {}
    pbar = tqdm(range(1, args.steps + 1), desc="Training with CMA-ES")
    for step in pbar:
        rng, rng_ask, rng_tell = jax.random.split(rng, 3)
        if step % args.eval_interval == 0 or step == 1:
            val_accuracy = evaluate_model_acc(model, test_loader, device)
        if step > args.warm:
            train_scheduler.step()
        if step % args.gd_interval == 0 or step == 1:
            # print("Before SGD fitness:", obj(model, batch, device, train=True))
            val_before = evaluate_model_acc(model, val_loader, device)
            gd_fe, gd_loss = train_on_gd(model, train_loader, optimizer, criterion, step=step, warmup_scheduler=warmup_scheduler, device=device)
            # print("After SGD fitness:", obj(model, batch, device, train=True))
            val_after = evaluate_model_acc(model, val_loader, device)
            if val_after >= val_before:
                gd_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()
                codebook, centers, log_sigmas, assignment = ubp_cluster(W=W_init, params=gd_params)
                W = len(centers)
                D = W * 2
                x0 = np.concatenate([centers, log_sigmas])
                solver = CMA_ES(population_size=popsize, solution=x0)
                # es_params = solver.default_params
                # es_params = es_params.replace(std_init=state.std.mean())
                state = solver.init(rng, x0, es_params)
            total_fe += gd_fe
            # else:
            #     model = build_model(model, D, total_weights, mu, codebook, state, weight_offsets)
            #     print("SGD did not improve the validation accuracy, skipping CMA-ES update")
    
        solutions, state = solver.ask(rng_ask, state, es_params)
        fitness_values = np.zeros(len(solutions))
        batch = next(train_iter)
        for i, solution in enumerate(solutions):
            model = build_model(model, W, total_weights, solution, codebook, state, weight_offsets)
            # Evaluate model
            f = obj(model, batch, device, train=True)
            weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()
            penalty = args.weight_decay * compute_l2_norm(weights)
            f += penalty
            fitness_values[i] = f # Minimize 
        total_fe += popsize
        # Update CMA-ES
        state, metrics = solver.tell(rng_tell, solutions, fitness_values, state, es_params)
        mu = solver.get_mean(state)
        model = build_model(model, W, total_weights, mu, codebook, state, weight_offsets)
        mean_fitness = obj(model, batch, device, train=True)
        min_fitness = np.min(fitness_values)
        average_fitness = np.mean(fitness_values)
        state_wandb = {
                "test accuracy": val_accuracy,
                "step": step,
                "min fitness": min_fitness,
                "mean fitness": mean_fitness,
                "average fitness": average_fitness,
                "sigma": np.mean(state.std),
                "function evaluations": total_fe,
                "LR": optimizer.param_groups[0]['lr'],
                "D": D,
                "W": W
            }
        wandb.log(state_wandb)
        # Logging
        pbar.set_postfix({"D": D, "fitness": f"{min_fitness:.4f}", "fe": total_fe})
        # print(f"Step {step + 1}, D:{D}, Fitness: {np.min(fitness_values)}, sigma: {np.mean(state.std)}")

        if np.mean(state.std) == 0  or np.isnan(np.mean(state.std)):
            print("sigma reached to 0, halting the optimization...")
            break

    mu = solver.get_mean(state)
    model = build_model(model, D, total_weights, mu, codebook, state, weight_offsets)
    # Final Evaluation
    final_accuracy = evaluate_model_acc(model, test_loader, device)
    print(f"Final Test Accuracy: {final_accuracy}%")
    wandb.log({"final_test_accuracy": final_accuracy})

    save_model(model, f"GF-CMA_ES-{args.net}-{args.dataset}-LR{args.lr_init}-W_init{args.w_init}-sigma_init{args.sigma_init}-batch_size{args.batch_size}-steps{args.steps}-warm{args.warm}", wandb)

    # Finish wandb run
    wandb.finish()

# Run Training
def parse_args():
    """
    Parse command line arguments
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR using CMA-ES')
    
    # Training parameters
    parser.add_argument('--warm', type=int, default=1,
                      help='Warmup steps for SGD (default: 1)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'],
                      help='Dataset to use (cifar10 or cifar100)')
    
    # CMA-ES parameters
    parser.add_argument('--w_init', type=int, default=256,
                      help='Codebook size (default: 2^8=256)')
    parser.add_argument('--steps', type=int, default=200,
                      help='Number of steps for CMA-ES (default: 200)')
    parser.add_argument('--popsize', type=int, default=100,
                      help='Population size for CMA-ES (default: 100)')
    parser.add_argument('--sigma_init', type=float, default=0.001,
                      help='Sigma initialization for CMA-ES (default: 0.001)')
    parser.add_argument('--lr_init', type=float, default=0.1,
                      help='Learning rate for SGD (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                      help='Weight decay for SGD (default: 0.001)')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                      help='Learning rate decay for SGD (default: 0.9)')
    parser.add_argument('--eval_interval', type=int, default=10,
                      help='Evaluation interval for CMA-ES (default: 100)')
    parser.add_argument('--gd_interval', type=int, default=10,
                      help='Evaluation interval for CMA-ES (default: 100)')
    parser.add_argument('--objective', type=str, default='f1',
                      choices=['ce', 'acc', 'f1'],
                      help='Objective function to optimize (default: f1)')
    
    # Training parameters
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for training (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for training (default: 128)')
    
    # Model parameters  
    parser.add_argument('--net', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34', 'resnet50', 'mnist30k', 'mnist500k', 'mnist3m', 'cifar300k', 'cifar900k', 'cifar8m'],
                      help='ResNet model architecture')
    
    # Logging parameters
    # parser.add_argument('--wandb_project', type=str, default='resnet-cmaes',
    #                   help='Weights & Biases project name')
    # parser.add_argument('--wandb_entity', type=str, default=None,
    #                   help='Weights & Biases entity name')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
# train_with_cma_es(W=int(2 ** 8), generations=10000, population_size=20, device='cuda')