class LeNetMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # First conv layer: 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # Second conv layer: 6 input channels, 16 output channels, 5x5 kernel  
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # First fully connected layer
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # Second fully connected layer
        self.fc2 = nn.Linear(120, 84)
        # Output layer
        self.fc3 = nn.Linear(84, 10)
        # Activation function
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.act(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        
        # Second conv block
        x = self.act(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        
        return x
    

class LeNetCIFAR(nn.Module):
    def __init__(self, num_channels=1, num_classes=10):
        super(LeNetCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x