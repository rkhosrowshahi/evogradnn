from models.resnets import ResNet18, ResNet34, ResNet50
import torch, time, yaml, os
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

def get_balanced_indices(dataset, num_classes):
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
        test_size=1000,  # 10% validation
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
        # val_indices = torch.randperm(50000)[:1000]
        # train_indices = torch.randperm(50000)[1000:]
        val_indices, train_indices = get_balanced_indices(train_dataset, num_classes)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        # val_indices = torch.randperm(50000)[:1000]
        # train_indices = torch.randperm(50000)[1000:]
        val_indices, train_indices = get_balanced_indices(train_dataset, num_classes)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    elif dataset == 'mnist':
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # val_indices = torch.randperm(60000)[:1000]
        # train_indices = torch.randperm(60000)[1000:]
        val_indices, train_indices = get_balanced_indices(train_dataset, num_classes)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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