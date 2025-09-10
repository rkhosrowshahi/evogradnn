import argparse
import os
import random
import sys
from typing import Dict, List, Tuple
import numpy as np
import optax
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import scienceplots
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Subset
from evosax.algorithms.distribution_based import distribution_based_algorithms
from evosax.algorithms.population_based import population_based_algorithms
from evosax.core.fitness_shaping import *
from scipy.stats import norm
import wandb

# --------------------- Weights & Biases Functions ---------------------

def init_wandb(project_name: str, run_name: str = None, config: dict = None, group: str = None, tags: list = None) -> None:
    """Initialize Weights & Biases logging.
    
    Args:
        project_name: Name of the W&B project
        run_name: Optional name for the run
        config: Dictionary of hyperparameters and configuration
        group: Optional group name for organizing runs
        tags: Optional list of tags for the run
    """
    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        group=group,
        tags=tags,
        reinit=True
    )

def log_metrics(metrics_dict: dict, step: int = None) -> None:
    """Log metrics to Weights & Biases.
    
    Args:
        metrics_dict: Dictionary of metric names and values
        step: Optional step number for the metrics
    """
    if wandb.run is not None:
        wandb.log(metrics_dict, step=step)

def log_evaluation_metrics(epoch: int, metrics: dict = None) -> None:
    """Log training and testing metrics to W&B.
    
    Args:
        epoch: Current epoch number
        metrics: Dictionary of metric names and values
        metrics: Optional dictionary of additional metrics
    """
    log_dict = {
        'Epoch': epoch,
    }
    
    if metrics:
        log_dict.update(metrics)
    
    log_metrics(log_dict)

def log_evolution_metrics(epoch: int, batch: int, metrics: dict = None) -> None:
    """Log evolution algorithm metrics to W&B.
    
    Args:
        epoch: Current epoch number
        metrics: Optional dictionary of additional metrics
    """
    log_dict = {
        'Epoch': epoch,
        'Batch': batch,
    }
    
    if metrics:
        log_dict.update(metrics)
    
    log_metrics(log_dict)

def finish_wandb_run(log_dict: dict = None) -> None:
    """Finish the current W&B run."""
    if log_dict is not None:
        wandb.log(log_dict)
    if wandb.run is not None:
        wandb.finish()

# --------------------- Model Save/Load Functions ---------------------

def save_model(model: nn.Module, name: str, wandb_run=None) -> None:
    """Save model to Weights & Biases.
    
    Args:
        model: The PyTorch model to save
        name: Name to save the model as
        wandb_run: Weights & Biases run instance (optional)
    """
    if wandb_run is None:
        wandb_run = wandb
    
    if wandb_run.run is not None:
        save_path = os.path.join(wandb_run.run.dir, f'{name}.pt')
        torch.save(model.state_dict(), save_path)
        print(f'Model saved to: {save_path}')
        
        # Also save as wandb artifact
        artifact = wandb.Artifact(name, type='model')
        artifact.add_file(save_path)
        wandb_run.log_artifact(artifact)

def load_model(name: str) -> Dict:
    """Load a saved model.
    
    Args:
        name: Path to the saved model
        
    Returns:
        The loaded model state dictionary
    """
    return torch.load(name, map_location=torch.device('cpu'))

# --------------------- Dataset Functions ---------------------

def get_balanced_indices(dataset, split_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get balanced indices for train/validation split using stratification.
    
    Args:
        dataset: PyTorch dataset
        split_size: Size of the validation split
        
    Returns:
        Tuple of (validation indices, training indices)
    """
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    indices = range(len(dataset))
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=split_size,
        stratify=all_labels,
        random_state=42,
        shuffle=True
    )
    
    return torch.tensor(val_idx), torch.tensor(train_idx)

def get_dataset(dataset: str, validation_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load and prepare dataset with train/val/test splits.
    
    Args:
        dataset: Name of dataset ('cifar100', 'cifar10', or 'mnist')
        validation_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes)
        
    Raises:
        ValueError: If dataset name is not supported
    """
    if dataset == 'cifar100':
        num_classes = 100
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        dataset_class = torchvision.datasets.CIFAR100
        
    elif dataset == 'cifar10':
        num_classes = 10
        stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        dataset_class = torchvision.datasets.CIFAR10
        
    elif dataset == 'mnist':
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_class = torchvision.datasets.MNIST
        transform_train = transform_test = transform
        
    elif dataset == 'fashion':
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        dataset_class = torchvision.datasets.FashionMNIST
        transform_train = transform_test = transform
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Load datasets
    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    
    # Create validation split
    val_indices, train_indices = get_balanced_indices(train_dataset, split_size=int(validation_split * len(train_dataset)))

    # Create subsets from original dataset
    val_dataset = Subset(train_dataset, val_indices)
    train_dataset = Subset(train_dataset, train_indices)
    
    return train_dataset, val_dataset, test_dataset, num_classes

def create_balanced_dataset(dataset, num_classes: int, samples_per_class: int = None) -> Subset:
    """Create a balanced dataset with equal samples per class.
    
    Args:
        dataset: PyTorch dataset (can be a Subset or regular dataset)
        num_classes: Number of classes in the dataset
        samples_per_class: Number of samples per class. If None, uses minimum class count
        
    Returns:
        Subset: Balanced dataset with equal samples per class
    """
    # Get all labels from the dataset
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label)
    
    # Group indices by class
    class_indices = {}
    for idx, label in enumerate(all_labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Determine samples per class
    if samples_per_class is None:
        # Use the minimum class count to ensure balance
        samples_per_class = min(len(indices) for indices in class_indices.values())
    
    # Sample balanced indices
    balanced_indices = []
    for class_label in range(num_classes):
        if class_label in class_indices:
            class_idx = class_indices[class_label]
            # Randomly sample from this class
            np.random.shuffle(class_idx)
            selected_indices = class_idx[:samples_per_class]
            balanced_indices.extend(selected_indices)
    
    # Shuffle the final indices to mix classes
    np.random.shuffle(balanced_indices)
    
    return Subset(dataset, balanced_indices)

# --------------------- Training Utilities ---------------------

class WarmUpLR(_LRScheduler):
    """Warmup learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        total_iters: Total iterations for warmup phase
        last_epoch: The index of last epoch
    """
    def __init__(self, optimizer, total_iters: int, last_epoch: int = -1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on warmup progress."""
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------- Model Evaluation Functions ---------------------

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    train: bool = False,
    device: str = 'cuda'
) -> float:
    """Evaluate model accuracy on a dataset.
    
    Args:
        model: Neural network model
        data_loader: DataLoader containing the dataset
        device: Device to run evaluation on
        train: If True, only evaluate on one batch
        
    Returns:
        Classification accuracy as percentage
    """
    model.eval()
    model.to(device)
    loss = 0
    correct = 0
    total = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if train:
                correct *= -1  # Negate for minimization in CMA-ES
                break
                
    return loss / num_batches, 100 * correct / total

def evaluate_model_acc(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    train: bool = False
) -> float:
    """Evaluate model accuracy on a dataset.
    
    Args:
        model: Neural network model
        data_loader: DataLoader containing the dataset
        device: Device to run evaluation on
        train: If True, only evaluate on one batch and negate for minimization
        
    Returns:
        Classification accuracy as percentage
    """
    model.eval()
    model.to(device)
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
                correct *= -1  # Negate for minimization in optimization
                break
                
    accuracy = 100 * correct / total
    return accuracy

def evaluate_model_ce(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    train: bool = False
) -> float:
    """Evaluate model using cross entropy loss.
    
    Args:
        model: Neural network model
        data_loader: DataLoader containing the dataset
        device: Device to run evaluation on
        train: If True, only evaluate on one batch
        
    Returns:
        Average cross entropy loss
    """
    model.eval()
    model.to(device)
    loss = 0
    total_batch = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += nn.functional.cross_entropy(outputs, labels).item()
            total_batch += 1
            
            if train:
                break
                
    if np.isnan(loss) or np.isinf(loss):
        loss = 9.9e+21  # Handle numerical instability
        
    return loss

def evaluate_model_acc_single_batch(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    train: bool = False
) -> float:
    """Evaluate model accuracy on a single batch.
    
    Args:
        model: Neural network model
        batch: Tuple of (inputs, labels)
        device: Device to run evaluation on
        train: Unused parameter for API compatibility
        
    Returns:
        Negative classification accuracy as percentage (for minimization)
    """
    model.eval()
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        
    return -100 * correct / total

def evaluate_model_f1score_single_batch(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    train: bool = False
) -> float:
    """Evaluate model F1 score on a single batch.
    
    Args:
        model: Neural network model
        batch: Tuple of (inputs, labels)
        device: Device to run evaluation on
        train: Unused parameter for API compatibility
        
    Returns:
        Negative F1 score (for minimization)
    """
    model.eval()
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
    f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
    return -f1

def evaluate_model_ce_single_batch(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    train: bool = False
) -> float:
    """Evaluate model using cross entropy loss on a single batch.
    
    Args:
        model: Neural network model
        batch: Tuple of (inputs, labels)
        device: Device to run evaluation on
        train: Unused parameter for API compatibility
        
    Returns:
        Cross entropy loss for the batch
    """
    model.eval()
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels).item()
        
    if np.isnan(loss) or np.isinf(loss):
        loss = 9.9e+21  # Handle numerical instability
        
    return loss

# --------------------- Distribution-based Strategy Functions ---------------------

def compute_l2_norm(x: np.ndarray) -> float:
    """Compute L2-norm of x_i.
    
    Args:
        x: Input array of shape (popsize, num_dims)
        
    Returns:
        Mean of squared values
    """
    return np.nanmean(x * x)

def build_model(
    model: nn.Module,
    W: int,
    total_weights: int,
    solution: np.ndarray,
    codebook: Dict[int, np.ndarray],
    state: Dict,
    weight_offsets: np.ndarray,
    device: str = 'cuda'
) -> nn.Module:
    """Build model using solution parameters from distribution-based strategy.
    
    Args:
        model: Base neural network model
        W: Number of components
        total_weights: Total number of parameters
        solution: Solution vector from distribution-based strategy
        codebook: Dictionary mapping components to parameter indices
        state: Distribution-based strategy state
        weight_offsets: Random offsets for parameters
        device: Device to place model on
        
    Returns:
        Updated model with new parameters
    """
    solution = np.array(solution)
    means = solution[:W]
    log_sigmas = solution[W:]
    sigmas = np.exp(log_sigmas)

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

    torch.nn.utils.vector_to_parameters(params, model.parameters())
    return model

def train_on_gd(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    step: int = 0,
    warmup_scheduler: WarmUpLR = None,
    args = None,
    device: str = 'cuda'
) -> Tuple[int, float]:
    """Train model using gradient descent.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        criterion: Loss function
        step: Current training step
        warmup_scheduler: Optional warmup learning rate scheduler
        args: Training arguments
        device: Device to run training on
        
    Returns:
        Tuple of (total function evaluations, average loss)
    """
    model.train()
    running_loss = 0.0
    total_fe = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total_fe += 1

        if step <= args.warm and warmup_scheduler is not None:
            warmup_scheduler.step()
    
    return total_fe, running_loss / total_fe

# --------------------- Clustering Functions ---------------------

def ubp_cluster(
    W: int,
    params: np.ndarray
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Uniform bin partitioning clustering.
    
    Args:
        W: Number of bins/clusters
        params: Parameters to cluster
        
    Returns:
        Tuple of (codebook, centers, log_sigmas, bin_indices)
    """
    min_val = params.min()
    max_val = params.max()
    bins = np.linspace(min_val, max_val, W)
    bin_indices = np.digitize(params, bins) - 1
    
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
        counter += 1
        
    return codebook, np.array(centers), np.array(log_sigmas), bin_indices

def random_codebook_initialization(W_init: int, total_weights: int) -> Dict[int, np.ndarray]:
    """Initialize random codebook for clustering.
    
    Args:
        W_init: Number of initial clusters
        total_weights: Total number of parameters
        
    Returns:
        Codebook mapping cluster indices to parameter indices
    """
    weight_indices = np.arange(total_weights)
    np.random.shuffle(weight_indices)
    
    codebook = {}
    d = np.random.dirichlet(np.ones(W_init))
    start_idx = 0
    
    for key in range(W_init):
        size = np.ceil(d[key] * total_weights).astype(int)
        end_idx = start_idx + size
        indices = weight_indices[start_idx:end_idx]
        
        if len(indices) == 0:
            indices = weight_indices[start_idx:end_idx+1]
            
        codebook[key] = indices
        weight_indices[indices] = np.full(len(indices), key)
        start_idx = end_idx
        
    return codebook

# --------------------- Utility Classes ---------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class Logger:
    """Custom logger that writes to both terminal and file."""
    
    def __init__(self, log_file: str):
        """Initialize logger.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()

# --------------------- Training and Visualization Functions ---------------------

def sgd_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    steps: int = 5,
    countfe: int = 0,
    lr: float = 1e-2,
    device: str = 'cuda'
) -> None:
    """Fine-tune model using SGD.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        steps: Number of optimization steps
        lr: Learning rate
        device: Device to run training on
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= steps:
            break
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        countfe += 1

def f1_score_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate F1 score error for model output.
    
    Args:
        output: Model output logits
        target: Ground truth labels
        
    Returns:
        Negative F1 score (for minimization)
    """
    _, predicted = torch.max(output.data, 1)
    predicted = predicted.cpu().numpy()
    target = target.cpu().numpy()
    
    f1 = -f1_score(target, predicted, average='macro')
    return torch.tensor(f1, device=output.device)

def plot_convergence(acc_tracker: Dict[int, float], save_path: str) -> None:
    """Plot convergence curve of accuracy over iterations.
    
    Args:
        acc_tracker: Dictionary mapping iteration to accuracy
        save_path: Directory to save plot
    """
    with plt.style.context(['science', 'no-latex']):
        plt.figure(figsize=(10, 5))
        plt.plot(list(acc_tracker.keys()), list(acc_tracker.values()))
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy ($\\%$)')
        plt.title('Convergence')
        plt.grid(True)
    plt.savefig(f'{save_path}/convergence.pdf')
    plt.close()

# --------------------- Parameter Management Functions ---------------------

def get_param_shapes(model: nn.Module) -> List[torch.Size]:
    """Get shapes of all trainable parameters in model.
    
    Args:
        model: Neural network model
        
    Returns:
        List of parameter shapes
    """
    return [param.shape for param in model.parameters() if param.requires_grad]

def params_to_vector(params: List[torch.Tensor], to_numpy: bool = False) -> torch.Tensor:
    """Flatten parameters into single vector.
    
    Args:
        params: List of parameter tensors
        
    Returns:
        Flattened parameter vector
    """
    params_vector = torch.nn.utils.parameters_to_vector(params).detach()
    if to_numpy:
        return params_vector.cpu().numpy()
    return params_vector

def assign_flat_params(model: nn.Module, params: torch.Tensor) -> None:
    """Assign flattened parameters back to model.
    
    Args:
        model: Neural network model
        params: Flattened parameter vector
    """
    parameters = model.parameters()
    # Ensure vec of type Tensor
    if not isinstance(params, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(flat_params)}")

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        if not param.requires_grad:
            continue
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = params[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def freeze_bn(model: nn.Module) -> None:
    # for m in model.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.track_running_stats = False
    #         m.running_mean = None
    #         m.running_var = None

    # for name, param in model.named_parameters():
    #         if "bn" in name:
    #             param.requires_grad = False

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

def unfreeze_bn(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "bn" in name:
            param.requires_grad = True



def distribution_based_strategy_init(key: jax.random.PRNGKey, strategy: str, x0: np.ndarray, steps: int, args: argparse.Namespace) -> None:
    if strategy == 'CMA_ES':
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0,
        )
        es_params = es.default_params.replace(
            std_init=args.std,
            std_min=1e-6, 
            std_max=1e1
        )
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'SV_CMA_ES':
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize//5, 
            num_populations=5,
            solution=x0,
        )
        es_params = es.default_params.replace(std_init=args.std, std_min=1e-6, std_max=1e1)
        means = np.random.normal(x0, args.std, (5, x0.shape[0]))
        es_state = es.init(key=key, means=means, params=es_params)
    elif strategy == 'SimpleES':
        lr_schedule = optax.exponential_decay(
            init_value=0.1,
            transition_steps=steps,
            decay_rate=0.1,
        )
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0,
            optimizer=optax.sgd(learning_rate=lr_schedule),
        )
        es_params = es.default_params.replace(
            std_init=args.std,
        )
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'Open_ES':
        lr_schedule = optax.cosine_decay_schedule(
            init_value=1e-3,
            decay_steps=steps,
            alpha=1e-2,
        )
        std_schedule = optax.cosine_decay_schedule(
            init_value=args.std,
            decay_steps=steps,
            alpha=1e-2,
        )
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0,
            optimizer=optax.adam(learning_rate=lr_schedule),
            std_schedule=std_schedule,
            use_antithetic_sampling=True,
        )
        es_params = es.default_params
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'SV_Open_ES':
        lr_schedule = optax.exponential_decay(
            init_value=0.001,
            transition_steps=steps * args.epochs,
            decay_rate=0.2,
        )
        std_schedule = optax.cosine_decay_schedule(
            init_value=args.std,
            decay_steps=steps * args.epochs,
            alpha=1e-4,
        )
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize//5, 
            num_populations=5,
            solution=x0,
            optimizer=optax.sgd(learning_rate=0.001),
            std_schedule=std_schedule,
            use_antithetic_sampling=True,
        )
        es_params = es.default_params
        es_state = es.init(key=key, means=np.random.normal(x0, 0.1, (5, x0.shape[0])), params=es_params)
    elif strategy == 'xNES':
        lr_schedule = optax.exponential_decay(
            init_value=0.001,
            transition_steps=steps,
            decay_rate=0.1,
        )
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0,
            optimizer=optax.sgd(learning_rate=0.001),
        )
        es_params = es.default_params
        es_params = es_params.replace(
            std_init=args.std,
        )
        es_state = es.init(key=key, mean=x0, params=es_params)
    return es, es_params, es_state


def population_based_strategy_init(strategy: str, args: argparse.Namespace, x0: np.ndarray, steps: int) -> None:
    if strategy == 'DE':
        es = population_based_algorithms['DifferentialEvolution'](
            population_size=args.popsize, 
            solution=x0,
            num_diff=1,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params.replace(
            elitism=True,
            differential_weight=0.5,
            crossover_rate=0.9,
        )
    elif strategy == 'PSO':
        es = population_based_algorithms['PSO'](
            population_size=args.popsize, 
            solution=x0,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params.replace(
            inertia_coeff=0.729,
            cognitive_coeff=1.49445,
            social_coeff=1.49445,
        )
    elif strategy == 'DiffusionEvolution':
        es = population_based_algorithms['DiffusionEvolution'](
            population_size=args.popsize, 
            solution=x0,
            num_generations=steps,
            # num_latent_dims=2,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params.replace(
            std_m=0.01,
            scale_factor=0.1,
        )
    return es, es_params


def fitness(z, model, base_params, decoder, batch, loss_fn, device, alpha):
    """Compute fitness of a solution.
    
    Args:
        z: Latent vector to evaluate.
        model: Model to evaluate.
        base_params: Base parameters to use.
        decoder: Decoder to use.
        batch: Batch of data.
        loss_fn: Loss function.
        device: Device to use.
        
    Returns:
        float: Fitness value (loss).
    """
    model.eval()
    decoder.apply(model=model, z=z, base_params=base_params, alpha=alpha)
    total_loss = 0.0
    
    with torch.no_grad():
        x, y = batch
        x, y = x.to(device), y.to(device)
        output = model(x)
        total_loss += loss_fn(output, y).item()
        
    return total_loss


def plot_trajectory(param_trajectory, sample_trajectory=None, save_path=None):
    # Convert trajectory to numpy array
    param_trajectory = np.array(param_trajectory)

    # Fit PCA and transform trajectory
    pca = PCA(n_components=2)
    pca.fit(param_trajectory)
    trajectory_2d = pca.transform(param_trajectory)

    if sample_trajectory is not None:
        sample_trajectory_2d = pca.transform(sample_trajectory)

    # Plot the trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'b-', label='Parameter Trajectory')
    plt.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                c=range(1, len(trajectory_2d)+1), cmap='viridis', 
                label='Training Progress', zorder=20)
    if sample_trajectory is not None:
        plt.scatter(sample_trajectory_2d[:, 0], sample_trajectory_2d[:, 1],
                    c='red', alpha=0.5, 
                    zorder=10, label='Gaussian Sampling on Parameter Trajectory')
    plt.colorbar(label='Training Progress')
    plt.title('Parameter Trajectory During Training (PCA Projection)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_3d_trajectory(param_trajectory, loss_trajectory, sample_trajectory=None, sample_loss_trajectory=None, save_path=None):
    # Convert trajectory to numpy array
    param_trajectory = np.array(param_trajectory)
    loss_trajectory = np.array(loss_trajectory)
    
    # Ensure positive values for log scale
    loss_trajectory = np.abs(loss_trajectory) + 1e-10  # Add small epsilon to avoid zero
    if sample_loss_trajectory is not None:
        sample_loss_trajectory = np.abs(sample_loss_trajectory) + 1e-10

    # Fit PCA and transform trajectory
    pca = PCA(n_components=2)
    pca.fit(param_trajectory)
    trajectory_2d = pca.transform(param_trajectory)

    if sample_trajectory is not None:
        sample_trajectory_2d = pca.transform(sample_trajectory)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    scatter = ax.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], loss_trajectory,
                        c=range(len(trajectory_2d)), cmap='viridis', zorder=20, s=100,
                        label='Training Progress')
    ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], loss_trajectory, 'b-', 
            alpha=0.3, zorder=15, label='Parameter Trajectory')

    if sample_trajectory is not None:
        ax.scatter(sample_trajectory_2d[:, 0], sample_trajectory_2d[:, 1], sample_loss_trajectory, 
                   c='red', alpha=0.5, zorder=10, s=10,
                   label='Gaussian Sampling on Parameter Trajectory')

    # Add colorbar
    plt.colorbar(scatter, label='Training Progress')

    # Set labels and title
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Loss Value (log scale)')
    ax.set_title('Parameter Trajectory During Training (3D PCA Projection)')

    # Set z-axis to log scale
    ax.set_zscale('log')

    # Add legend
    ax.legend()

    # Set the view angle
    ax.view_init(elev=30, azim=45)

    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_param_distribution(x, save_path):
    
    # Plot histogram of x
    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(x, bins=100, density=True, alpha=0.6, color='g', label='Parameters')

    # Plot the theoretical normal distribution curve
    mu, std = np.mean(x), np.std(x)
    pdf = norm.pdf(bins, mu, std)
    plt.plot(bins, pdf, 'k--', linewidth=2, label='Normal PDF')

    plt.title('Parameter distribution vs. Normal distribution')
    plt.xlabel(f'$\\theta$')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}")
    plt.close()


class CosineAnnealingScheduler:
    def __init__(self, eta_max, eta_min, T_max, T_mult=1):
        """
        Cosine Annealing Learning Rate Scheduler.
        
        Args:
            eta_max (float): Maximum learning rate.
            eta_min (float): Minimum learning rate.
            T_max (int): Number of steps per half-cycle (cosine decay period).
            T_mult (float): Multiplier for T_max after each restart (default: 1, no increase).
        """
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_max = T_max
        self.T_mult = T_mult
        self.current_step = 0
        self.current_T_max = T_max
        self.cycle = 0

    def get_lr(self):
        """
        Compute the learning rate for the current step.
        
        Returns:
            float: Current learning rate.
        """
        cosine_decay = 0.5 * (1 + np.cos(np.pi * (self.current_step % self.current_T_max) / self.current_T_max))
        lr = self.eta_min + (self.eta_max - self.eta_min) * cosine_decay
        return lr

    def step(self):
        """
        Increment the step and update the scheduler.
        Handles warm restarts if T_mult > 1.
        """
        self.current_step += 1
        if self.current_step % self.current_T_max == 0:
            self.cycle += 1
            self.current_T_max = self.T_max * (self.T_mult ** self.cycle)
            self.current_step = 0