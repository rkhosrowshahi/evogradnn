# Import CIFAR10 model variants
from .CIFAR10 import CIFAR300K, CIFAR900K, CIFAR8M
# Import MNIST model variants
from .MNIST import MNIST30K, MNIST500K, MNIST3M
# Import ResNet architectures
# from .resnets import ResNet18, ResNet34, ResNet50
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

def get_model(model_name, num_classes, device='cuda'):
    """
    Factory function to create and return a model based on the specified name.
    Args:
        model_name (str): Name of the model to instantiate
        num_classes (int): Number of output classes (only for ResNet models)
        device (str): Device to place the model on ('cuda' or 'cpu')
    Returns:
        model: Instantiated model on the specified device
    """
    # ResNet model variants
    if model_name == 'resnet18':
        model = resnet18(num_classes=num_classes).to(device)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes).to(device)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes).to(device)
    elif model_name == 'resnet101':
        model = resnet101(num_classes=num_classes).to(device)
    elif model_name == 'resnet152':
        model = resnet152(num_classes=num_classes).to(device)
    # CIFAR10 model variants
    elif model_name == 'cifar300k':
        model = CIFAR300K().to(device)
    elif model_name == 'cifar900k':
        model = CIFAR900K().to(device)
    elif model_name == 'cifar8m':
        model = CIFAR8M().to(device)
    # MNIST model variants
    elif model_name == 'mnist30k':
        model = MNIST30K().to(device)
    elif model_name == 'mnist500k':
        model = MNIST500K().to(device)
    elif model_name == 'mnist3m':
        model = MNIST3M().to(device)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model

# List of all available models for export
__all__ = ["CIFAR300K", "CIFAR900K", "CIFAR8M", "MNIST30K", "MNIST500K", "MNIST3M", "ResNet18", "ResNet34", "ResNet50"]
