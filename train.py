import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Import the architecture from model.py file
from model import TrashClassifierCNN

# Define transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()         
])

# Load dataset
dataset_path = 'dataset'
full_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms) # It deduces automatically the class labels from the folder names and puts number labels in the dataset

# Split data
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train_model():
    # Check for NVIDIA GPU (Windows/Linux), then Apple Silicon (Mac), otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model and send it to the device (GPU/CPU)
    num_classes = len(full_dataset.classes)
    model = TrashClassifierCNN(num_classes=num_classes).to(device)

    # Define the Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Adam is a very popular and efficient optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Number of times we go through the entire dataset
    num_epochs = 50

    print("Starting training loop...")
    print("-" * 30)

    # Training loop
    for epoch in range(num_epochs):
        model.train() # Set model to training mode 
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to the same device as the model (MPS/CPU)
            images, labels = images.to(device), labels.to(device)

            # Clear old gradients
            optimizer.zero_grad()
            
            # Forward pass (make a prediction)
            outputs = model(images)
            
            # Calculate loss (how wrong the prediction was)
            loss = criterion(outputs, labels)
            
            # Backward pass (calculate corrections)
            loss.backward()
            
            # Optimizer step (apply corrections)
            optimizer.step()

            running_loss += loss.item()

        # Print progress at the end of each epoch
        average_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed | Average Loss: {average_loss:.4f}")

    print("-" * 30)
    print("Training finished!")

    # We save the trained model so predict.py can use it later
    save_path = 'model.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")

if __name__ == "__main__":
    train_model()