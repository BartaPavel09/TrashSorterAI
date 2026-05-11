import torch.nn as nn
import torch.nn.functional as F

class TrashClassifierCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(TrashClassifierCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Pass image through convolution, then activation (ReLU), then pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the 2D images into a 1D list of numbers
        x = x.view(-1, 64 * 28 * 28)
        
        # Pass through the fully connected decision layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x