import torch.nn as nn
# Import CIFAR10 model variants
from .cifar10 import CIFAR300K, CIFAR900K, CIFAR8M
# Import MNIST model variants
from .mnist import MNIST30K, MNIST500K, MNIST3M
from .lenet import LeNetCIFAR, LeNetMNIST
# Import ResNet architectures
# from .resnets import ResNet18, ResNet34, ResNet50
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet import resnet20, resnet32, resnet56

def get_model(model_name, input_size, num_classes, device='cuda'):
    """
    Factory function to create and return a model based on the specified name.
    Args:
        model_name (str): Name of the model to instantiate
        input_size (int): Input image size (e.g., 32 for CIFAR, 224 for ImageNet)
        num_classes (int): Number of output classes
        device (str): Device to place the model on ('cuda' or 'cpu')
    Returns:
        model: Instantiated model on the specified device
    """
    # ResNet model variants
    if model_name == 'resnet18':
        model = resnet18(num_classes=num_classes) # For imagenet, [2, 2, 2, 2]
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes) # For imagenet, [3, 4, 6, 3]
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes) # For imagenet, [3, 4, 6, 3]
    elif model_name == 'resnet101':
        model = resnet101(num_classes=num_classes) # For imagenet, [3, 4, 23, 3]
    elif model_name == 'resnet152':
        model = resnet152(num_classes=num_classes) # For imagenet, [3, 8, 36, 3]
    elif model_name == 'resnet20':
        model = resnet20(num_classes=num_classes) # For CIFARs, [3, 3, 3]
    elif model_name == 'resnet32':
        model = resnet32(num_classes=num_classes) # For CIFARs, [5, 5, 5]
    elif model_name == 'resnet56':
        model = resnet56(num_classes=num_classes) # For CIFARs, [7, 7, 7]
    # CIFAR10 model variants
    elif model_name == 'cifar300k':
        model = CIFAR300K() # For CIFARs
    elif model_name == 'cifar900k':
        model = CIFAR900K() # For CIFARs
    elif model_name == 'cifar8m':
        model = CIFAR8M() # For CIFARs
    # MNIST model variants
    elif model_name == 'mnist30k':
        model = MNIST30K() # For MNISTs
    elif model_name == 'mnist500k':
        model = MNIST500K() # For MNISTs
    elif model_name == 'mnist3m':
        model = MNIST3M() # For MNISTs
    elif model_name == 'lenet':
        if input_size == 32:
            model = LeNetCIFAR() # For CIFARs
        elif input_size == 28:
            model = LeNetMNIST() # For MNISTs
        else:
            raise ValueError(f"Input size {input_size} not supported for LeNet")
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        if input_size == 32:
        # Modify first conv layer for CIFAR (32x32) instead of ImageNet (224x224)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()  # Remove maxpool for CIFAR
                # Modify final layer for number of classes
                model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif input_size != 224:
            raise ValueError(f'Unsupported input size: {input_size}, only 32 or 224 are supported')
    return model.to(device)

# List of all available models for export
__all__ = ["CIFAR300K", "CIFAR900K", "CIFAR8M", 
            "MNIST30K", "MNIST500K", "MNIST3M", 
            "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "ResNet20", "ResNet32", "ResNet56", 
            "LeNetCIFAR", "LeNetMNIST"]
