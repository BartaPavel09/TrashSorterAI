import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()         
])

# Load the entire folder of images
# ImageFolder automatically reads subfolder names and turns them into labels
dataset_path = 'dataset'
full_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms)

# Split the data: 80% Training and 20% Testing
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size]
)

# Create data loaders 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    print(f"Found the following trash categories: {full_dataset.classes}")
    print(f"Total images found: {len(full_dataset)}")
    print(f"Images reserved for training: {len(train_dataset)}")
    print(f"Images reserved for testing: {len(test_dataset)}")
    
    # Take a single batch of 32 images to verify its mathematical shape
    images, labels = next(iter(train_loader))
    print(f"Shape of a single image batch: {images.shape}")