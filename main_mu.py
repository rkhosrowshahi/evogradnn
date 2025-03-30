import argparse
import itertools
import jax
from sklearn.metrics import f1_score
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

from utils import load_data, save_model
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
    means = solution
    c = 0.1
    sigma = [0]
    z = weight_offsets

    # Initialize parameter vector
    params = torch.zeros(total_weights, device=device)
    for k in range(W):
        indices = codebook[k]
        if len(indices) > 0:  # Handle empty components
            params[indices] = torch.full(
                size=(len(indices),),
                fill_value=torch.Tensor(means[k]),
                device=device
            )
    # Assign weights to model
    torch.nn.utils.vector_to_parameters(params, model.parameters())

    return model

def train_on_gd(model, train_loader, optimizer, criterion, device='cuda'):
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
    counter = 0
    codebook = {}
    for i in range(W):
        mask = np.where(bin_indices == i)[0]
        if len(mask) == 0:
            continue
        centers.append(params[mask].mean())
        bin_indices[mask] = counter
        codebook[counter] = mask

        counter+=1
    centers = np.array(centers)
    return codebook, centers, bin_indices

# CMA-ES Training Loop
def main(args):
    """
    Main training loop using CMA-ES
    Args:
        args: Command line arguments
    """
    # Initialize wandb logging
    wandb.init(project=f"{args.model}_{args.dataset}", config={
        "W": args.w,
        "steps": args.steps,
        "popsize": args.popsize,
        "device": args.device,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "model": args.model,
        "sigma_init": args.sigma_init,
        "lrate_init": args.lrate_init,
        "dataset": args.dataset,
        "weight_decay": args.weight_decay,
        "lrate_decay": args.lrate_decay,
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
    W = args.w
    popsize = args.popsize
    steps = args.steps
    
    train_loader, test_loader, num_classes = load_data(args.dataset, args.batch_size)

    model = get_model(args.model, num_classes, device)

    # print(evaluate(model, train_loader, device, train=True))
    total_weights = sum(p.numel() for p in model.parameters())
    initial_weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()

    weight_indices = np.arange(0, total_weights)
    np.random.shuffle(weight_indices)
    codebook = {}
    d = np.random.dirichlet(np.ones(W))
    start_idx = 0
    end_idx = 0
    for key in range(W):
        size = np.ceil(d[key] * total_weights).astype(int)
        start_idx = key * size
        end_idx = start_idx + size
        indices = weight_indices[start_idx:end_idx]
        if len(indices) == 0:
            indices = weight_indices[start_idx:end_idx+1]
        codebook[key] = indices
        weight_indices[indices] = np.full((len(indices), ), key)
         
    weight_offsets = torch.normal(mean=0.0, std=1.0, size=(total_weights,), device=device)

    # Initialize CMA-ES
    rng = jax.random.PRNGKey(0)
    x0 = np.concatenate([np.zeros(W)])
    solver = CMA_ES(population_size=popsize, solution=x0)
    es_params = solver.default_params
    es_params = es_params.replace(std_init=args.sigma_init)
    state = solver.init(rng, x0, es_params)
    state = state.replace(mean=x0)
    D = W


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lrate_init, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 1000, 2000], gamma=0.1)
    

    total_fe = 0
    train_iter = itertools.cycle(train_loader)
    # Training Loop
    state_wandb = {}
    for step in range(steps):
        rng, rng_ask, rng_tell = jax.random.split(rng, 3)
        batch = next(train_iter)
        if step % args.eval_interval == args.eval_interval - 1 or step == 0 :
            val_accuracy = evaluate_model_acc(model, test_loader, device)
            
        if step % args.gd_interval == args.gd_interval - 1 or step == 0:
            print("Before SGD fitness:", obj(model, batch, device, train=True))
            gd_fe, gd_loss = train_on_gd(model, train_loader, optimizer, criterion)
            print("After SGD fitness:", obj(model, batch, device, train=True))
            gd_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()
            codebook, centers, assignment = ubp_cluster(W, gd_params)
            D = len(centers)
            solver = CMA_ES(population_size=popsize, solution=centers)
            # es_params = solver.default_params
            es_params = es_params.replace(std_init=np.mean(state.std))
            state = solver.init(rng, centers, es_params)
            # state = state.replace(mean=params)
            total_fe += gd_fe

        solutions, state = solver.ask(rng_ask, state, es_params)
        fitness_values = np.zeros(len(solutions))

        
        for i, solution in enumerate(solutions):
            model = build_model(model, D, total_weights, solution, codebook, state, weight_offsets)
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
        model = build_model(model, D, total_weights, mu, codebook, state, weight_offsets)
        mean_fitness = obj(model, batch, device, train=True)
        min_fitness = np.min(fitness_values)
        average_fitness = np.mean(fitness_values)
        state_wandb = {
                "test_accuracy": val_accuracy,
                "step": step + 1,
                "min_fitness": min_fitness,
                "mean_fitness": mean_fitness,
                "average_fitness": average_fitness,
                "sigma": np.mean(state.std),
                "total_fe": total_fe,
                "gd_lr": optimizer.param_groups[0]['lr']
            }
        wandb.log(state_wandb)
        # Logging
        print(f"Step {step + 1}, D:{D}, Fitness: {np.min(fitness_values)}, sigma: {np.mean(state.std)}")

        if np.mean(state.std) == 0  or np.isnan(np.mean(state.std)):
            print("sigma reached to 0, halting the optimization...")
            break

        scheduler.step()

    mu = solver.get_mean(state)
    model = build_model(model, D, total_weights, mu, codebook, state, weight_offsets)
    # Final Evaluation
    final_accuracy = evaluate_model_acc(model, test_loader, device)
    print(f"Final Test Accuracy: {final_accuracy}%")
    wandb.log({"final_test_accuracy": final_accuracy})

    save_model(model, f"{args.model}_{args.dataset}_mu_0", wandb)

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
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'],
                      help='Dataset to use (cifar10 or cifar100)')
    
    # CMA-ES parameters
    parser.add_argument('--w', type=int, default=256,
                      help='Codebook size (default: 2^8=256)')
    parser.add_argument('--steps', type=int, default=10000,
                      help='Number of steps for CMA-ES (default: 10000)')
    parser.add_argument('--popsize', type=int, default=20,
                      help='Population size for CMA-ES (default: 20)')
    parser.add_argument('--sigma_init', type=float, default=0.001,
                      help='Sigma initialization for CMA-ES (default: 0.001)')
    parser.add_argument('--lrate_init', type=float, default=0.1,
                      help='Learning rate for SGD (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                      help='Weight decay for SGD (default: 0.001)')
    parser.add_argument('--lrate_decay', type=float, default=0.99,
                      help='Learning rate decay for SGD (default: 0.99)')
    parser.add_argument('--eval_interval', type=int, default=100,
                      help='Evaluation interval for CMA-ES (default: 100)')
    parser.add_argument('--gd_interval', type=int, default=100,
                      help='Evaluation interval for CMA-ES (default: 100)')
    parser.add_argument('--objective', type=str, default='ce',
                      choices=['ce', 'acc', 'f1'],
                      help='Objective function to optimize (default: ce)')
    
    # Training parameters
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for training (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training (default: 128)')
    
    # Model parameters  
    parser.add_argument('--model', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34', 'resnet50', 'mnist30k', 'mnist500k', 'mnist3m', 'cifar300k', 'cifar900k', 'cifar8m'],
                      help='ResNet model architecture')
    
    # Logging parameters
    parser.add_argument('--wandb_project', type=str, default='resnet-cmaes',
                      help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='Weights & Biases entity name')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
# train_with_cma_es(W=int(2 ** 8), generations=10000, population_size=20, device='cuda')