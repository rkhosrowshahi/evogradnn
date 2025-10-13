import torch
from torch import nn
import torch.nn.functional as F

class LeNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # -> 6x24x24
        self.pool = nn.MaxPool2d(2, 2) # -> halves spatial dims
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # -> 16x8x8

        # Flatten size after conv/pool: 16 * 4 * 4 = 256
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class LeNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetCIFAR, self).__init__()
        # Input: (3, 32, 32)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5) # -> (6, 28, 28)
        self.pool = nn.MaxPool2d(2, 2) # -> (6, 14, 14)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # -> (16, 10, 10)
        # Pool -> (16, 5, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 400 -> 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x