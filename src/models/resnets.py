import torch
import torch.nn as nn
import torchvision

# Define ResNet-18 Model
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)