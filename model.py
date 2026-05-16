import torch.nn as nn
import torch.nn.functional as F

class TrashClassifierCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(TrashClassifierCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) # This extracts the edges and basic features from the input image
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # This detects shapes and patterns from the features extracted by the first layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # This detects more complex patterns and features

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 512) # This is 28*28 because of the pooling layers reducing the spatial dimensions and 64 is the number of output channels from the last convolutional layer
        self.fc2 = nn.Linear(512, num_classes) # this is the output layer that gives us the probabilities for each class

    def forward(self, x):
        # ReLU transforms all negative values to zero and keeps positive values unchanged, which helps the model learn non-linear relationships in the data
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the 2D images into a 1D list of numbers
        x = x.view(-1, 64 * 28 * 28)
        
        # Pass through the fully connected decision layers
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) #this returns the 6 class probabilities for the input image
        
        return x