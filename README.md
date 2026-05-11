# TrashSorterAI

This project implements a Convolutional Neural Network (CNN) built from scratch using PyTorch. It is designed to classify images of waste into 6 specific categories (cardboard, glass, metal, paper, plastic, trash) to automate recycling sorting.

## 1. Dataset Setup
The model is trained on the TrashNet dataset. 
**Download the dataset here:** https://github.com/garythung/trashnet

After downloading, extract the archive and ensure the images are placed in a folder named `dataset` in the root directory of this project. The structure must look like this:
- `dataset/`
  - `cardboard/`
  - `glass/`
  - `metal/`
  - `paper/`
  - `plastic/`
  - `trash/`

## 2. Environment Setup
To run this project, you need Python installed. It is recommended to use a virtual environment.

1. Open a terminal in the project directory.
2. Install the required dependencies by running:
   `pip install -r requirements.txt`

## 3. Project Structure
- `model.py`: Contains the CNN architecture (built from scratch).
- `train.py`: Handles data loading, preprocessing, and the training loop.
- `predict.py`: Contains the logic to test the trained model on a single image.
- `model.pt`: The saved weights of the trained neural network.

## 4. How to Run

### Training the Model
To train the model from scratch, execute:
`python train.py`
*Note for macOS users: The training script automatically detects and utilizes Apple Silicon (MPS) for hardware acceleration if available.*

### Making a Prediction
To test the model on a new image:
1. Place an image in the project root directory and name it `img.jpg`.
2. Run the prediction script:
   `python predict.py`
The script will load the saved `model.pt` weights, analyze the image, and print the predicted category.
