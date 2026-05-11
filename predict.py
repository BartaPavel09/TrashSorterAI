import torch
from torchvision import transforms
from PIL import Image
import os

# Import the model architecture
from model import TrashClassifierCNN

# --- Configuration ---
# Update this path if you use another image name
IMAGE_PATH = 'img.jpg' 

# Map class numbers to human readable names
# This list must be in the exact alphabetical order used by ImageFolder
# (cardboard, glass, metal, paper, plastic, trash)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_and_transform_image(path):
    # Loads an image from a file and applies preprocessing for the CNN.
    try:
        # Open image using Pillow
        image = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Error opening image at {path}: {e}")
        return None

    # Apply the exact same transforms used during training setup
    prediction_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()          
    ])
    
    # Process image and add batch dimension (Batch size = 1)
    # Output shape becomes: [1, 3, 224, 224]
    image_tensor = prediction_transforms(image).unsqueeze(0)
    
    return image_tensor

def predict():
    #Performs inference on a single test image.
    
    if not os.path.exists(IMAGE_PATH):
        print(f"File not found: {IMAGE_PATH}. Please provide a valid image file.")
        return

    print(f"Processing image: {IMAGE_PATH}...")
    image_tensor = load_and_transform_image(IMAGE_PATH)
    
    if image_tensor is None:
        return

    # Initialize the model
    num_classes = len(CLASS_NAMES)
    model = TrashClassifierCNN(num_classes=num_classes)
    
    # Load the trained weights
    model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
    
    model.eval() # Set model to evaluation mode

    # Perform the forward pass
    with torch.no_grad():
        outputs = model(image_tensor)
        
    # Interpret results
    _, predicted_index = torch.max(outputs, 1)
    prediction_name = CLASS_NAMES[predicted_index.item()]
    
    print("-" * 30)
    print("Prediction result:")
    print("-" * 30)
    print(f"Model outputs (scores): {outputs.numpy()[0]}")
    print(f"Predicted class index: {predicted_index.item()}")
    print(f"Predicted category:    {prediction_name.upper()}")
    print("-" * 30)

if __name__ == "__main__":
    predict()