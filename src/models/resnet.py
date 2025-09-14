"""
Custom ResNet implementation for CIFAR datasets.

This module provides ResNet architectures optimized for CIFAR-10/100 datasets (32x32 images).
The implementation follows the original ResNet paper but with modifications for smaller input sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50, ResNet-101, and ResNet-152."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture optimized for CIFAR datasets."""
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # First layer: 3x3 conv with stride 1 (optimized for CIFAR 32x32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # Global average pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global average pooling
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # Final classifier
        out = self.fc(out)
        return out


def resnet32(weights=None, num_classes=10, **kwargs):
    """
    ResNet-32 for CIFAR datasets.
    
    This is a custom ResNet with 32 layers optimized for CIFAR-10/100.
    Architecture: 5 residual blocks in each of 3 stages (layer1, layer2, layer3).
    
    Args:
        weights: Pretrained weights (not supported for custom ResNet32)
        num_classes: Number of output classes
        **kwargs: Additional keyword arguments
        
    Returns:
        ResNet32 model
    """
    if weights is not None and weights != "DEFAULT":
        raise ValueError("Pretrained weights are not available for ResNet32. Use weights=None.")
    
    # ResNet-32: 5 blocks per layer, total = 1 + 3*(5*2) + 1 = 32 layers
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)
    return model


def resnet20(weights=None, num_classes=10, **kwargs):
    """
    ResNet-20 for CIFAR datasets.
    
    Args:
        weights: Pretrained weights (not supported for custom ResNet20)
        num_classes: Number of output classes
        **kwargs: Additional keyword arguments
        
    Returns:
        ResNet20 model
    """
    if weights is not None and weights != "DEFAULT":
        raise ValueError("Pretrained weights are not available for ResNet20. Use weights=None.")
    
    # ResNet-20: 3 blocks per layer, total = 1 + 3*(3*2) + 1 = 20 layers
    model = ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
    return model


def resnet56(weights=None, num_classes=10, **kwargs):
    """
    ResNet-56 for CIFAR datasets.
    
    Args:
        weights: Pretrained weights (not supported for custom ResNet56)
        num_classes: Number of output classes
        **kwargs: Additional keyword arguments
        
    Returns:
        ResNet56 model
    """
    if weights is not None and weights != "DEFAULT":
        raise ValueError("Pretrained weights are not available for ResNet56. Use weights=None.")
    
    # ResNet-56: 9 blocks per layer, total = 1 + 3*(9*2) + 1 = 56 layers
    model = ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)
    return model


# For compatibility with existing code that might expect these models
__all__ = ['ResNet', 'BasicBlock', 'Bottleneck', 'resnet20', 'resnet32', 'resnet56']
