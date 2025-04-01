import random
import numpy as np
from sklearn.metrics import f1_score
from models.resnets import ResNet18, ResNet34, ResNet50
import torch, time, yaml, os
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def save_model(model, name, wandb):
    """ Save model to Weights&Biases """
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'{name}.pt'))
    print('Model saved to Weights&Biases')
    print(os.path.join(wandb.run.dir, f'{name}.pt'))


def load_model(name):
    """ Load saved model """
    return torch.load(f'{name}', map_location=torch.device('cpu'))

def get_balanced_indices(dataset, split_size=1000):
    """ Get balanced indices using sklearn's train_test_split
    Args:
        dataset: PyTorch dataset
        num_classes: Number of classes in the dataset
    Returns:
        val_indices: Indices for validation set
        train_indices: Indices for training set
    """
    # Get all labels from the dataset
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    
    # Create indices array
    indices = range(len(dataset))
    
    # Split indices with stratification
    train_idx, val_idx = train_test_split(
        indices,
        test_size=split_size,  # 10% validation
        stratify=all_labels,
        random_state=42,
        shuffle=True
    )
    
    # Convert to tensors
    train_indices = torch.tensor(train_idx)
    val_indices = torch.tensor(val_idx)
    
    return val_indices, train_indices


def load_data(dataset, batch_size):
    if dataset == 'cifar100':
        num_classes = 100
        # Load CIFAR-100 Dataset
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Normalize with CIFAR-100 stats
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Normalize with CIFAR-100 stats
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        val_indices, train_indices = get_balanced_indices(train_dataset, split_size=1000)
        val_loader = DataLoader(Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset == 'cifar10':
        num_classes = 10
        # Load CIFAR-10 Dataset
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # Normalize with CIFAR-10 stats
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # Normalize with CIFAR-10 stats
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        val_indices, train_indices = get_balanced_indices(train_dataset, split_size=1000)
        val_loader = DataLoader(Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    elif dataset == 'mnist':
        num_classes = 10
        transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        val_indices, train_indices = get_balanced_indices(train_dataset, split_size=1000)
        val_loader = DataLoader(Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    return train_loader, val_loader, test_loader, num_classes



class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



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

def build_model(model, W, total_weights, solution, codebook, state, weight_offsets, device='cuda'):
    """
    Build model using the solution parameters from distribution based strategy
    Args:
        model: Base neural network model
        W: Number of components
        total_weights: Total number of parameters
        solution: Solution vector from distribution based strategy
        codebook: Dictionary mapping components to parameter indices
        state: Distribution based strategy state
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
            ) # * weight_offsets[indices]
    # Assign weights to model
    torch.nn.utils.vector_to_parameters(params, model.parameters())

    return model

def train_on_gd(model, train_loader, optimizer, criterion, step=0, warmup_scheduler=None, args=None, device='cuda'):
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
        
        # Backward and optimize
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
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
