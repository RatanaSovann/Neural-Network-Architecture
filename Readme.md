#  Neural Network Architecture (Multi Layered Perceptron)

<p align="center"><img width="358" height="328" alt="image" src="https://github.com/user-attachments/assets/18c90c30-1c69-4776-b93a-ceabac0718e5" /><p align="center">

## Objective
The main goal is to build, evaluate, and refine a neural network architecture that achieves at least 80% accuracy on the Japanese MNIST dataset, without relying on convolutional layers. Three experiments are conducted, each focusing on different model configurations or hyperparameters, and analyze their impact on both training and validation performance.

## Dataset Used
The custom Neural Network is trained on  the Japanese MNIST dataset which composed of 70,000 images of handwritten Hiragana characters, each belonging to one of ten different classes. Every image has dimensions of 28×28 pixels, which are flattened into vectors of size 784 for input into our fully connected neural network.

<p align="center"><img width="528" height="227" alt="image" src="https://github.com/user-attachments/assets/a67d280d-a5e0-4539-b708-7b65c2c30960" /><p align="center">

## Data Preperation
````python
#1. Import Required Packages
from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import requests
from tqdm import tqdm
import os.path


#2. Download Dataset
# Mount Google Drive
drive.mount('/content/gdrive')

# Create a working folder & set working directory
%cd '/content/gdrive/MyDrive/DL_ASG_1'

# 3. Download Data
def download_file(url):
    path = url.split('/')[-1]
    if os.path.isfile(path):
        print (f"{path} already exists")
    else:
      r = requests.get(url, stream=True)
      with open(path, 'wb') as f:
          total_length = int(r.headers.get('content-length'))
          print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))
          for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
              if chunk:
                  f.write(chunk)

url_list = [
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz'
]

for url in url_list:
    download_file(url)

````
The pre-processing steps follows: 
- Represent the data with the following dimensions (60000, 28, 28, 1).
- Both training and testing sets are converted to float32 (necessary for PyTorch).
- Since the images are in greyscale the dataset for each image contains pixels with values ranging from 0 to 255. These values are standardized to a new range of 0 to 1 for modelling. They are then converted to tensors using torch.tensor and loaded into a Dataloader for batch training.
- The data are split into 80/20 for training and validation.

````python
# Create a function that loads a .npz file using numpy and return the content of the arr_0 key
def load(f):
    return np.load(f)['arr_0']

# Load files into their respective variables
x_train = load('kmnist-train-imgs.npz')
x_test = load('kmnist-test-imgs.npz')
y_train = load('kmnist-train-labels.npz')
y_test =load('kmnist-test-labels.npz')

# Reshape the images from the training and testing set to have the channel dimension last. The dimensions should be: (row_number, height, width, channel)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Cast `x_train` and `x_test` into `float32` decimals
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Standardise the images of the training and testing sets. Originally each image contains pixels with value ranging from 0 to 255. after standardisation, the new value range should be from 0 to 1.
x_train = x_train / 255.0
x_test = x_test / 255.0

#  Create a variable called `num_classes` that will take the value 10 which corresponds to the number of classes for the target variable
num_classes = 10

# Convert the target variable for the training and testing sets to a binary class matrix of dimension (rows, num_classes).
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

````
Convert data into tensors. We use Google Collab GPU if available, as it processess tensors faster.

````python
# Create a variable called `device` that will automatically select a GPU if available. Otherwise it will default to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device = device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device = device)
X_test_tensor = torch.tensor(x_test, dtype=torch.float32, device = device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device = device)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

````
## Define Neural Networks Architecture
This MLP is a simple neural network that takes an input (like an image), flattens it into a list of numbers, and processes it step by step to make a prediction:

Flatten input → converts the image into a vector of numbers
First layer (fc1) → learns patterns from the input
ReLU activation → keeps important signals, removes weak ones
Dropout → randomly drops some neurons to prevent overfitting
Final layer (fc2) → produces output scores for each class (prediction)

````python
# Set seed in PyTorch to reproduce results
torch.manual_seed(0)

class MLP(nn.Module):    # Creating a class called MLP that inherits nn.Module
  def __init__(self, input_size, hidden_size, output_size):    # Initialize the class
      super(MLP,self).__init__()
      self.fc1 = nn.Linear(input_size,hidden_size) # Fully connected Layer1 (Input to hidden layer)
      self.relu = nn.ReLU() # Calling Rectified Linear Unit
      self.dropout = nn.Dropout(0.5)
      self.fc2 = nn.Linear(hidden_size,output_size) # Fully connected Layer2 (Hidden to output layer)

  def forward(self,x):
    x = x.view(x.size(0), -1) # Flatten image
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return x

````
````python
# Set parameters
input_size = 28*28
hidden_size = 128
output_size = 10

model = MLP(input_size, hidden_size, output_size).to(device)

````
<img width="911" height="155" alt="image" src="https://github.com/user-attachments/assets/4b0b3be3-c739-4178-85d3-ff4528e80490" />

## Training Neural Network 

This setup is designed to effectively train a multi-class classification model by combining a reliable optimizer, an appropriate loss function, and a simple evaluation metric.
- The Adam optimizer is used because it adaptively adjusts learning rates for each parameter, allowing the model to learn efficiently and converge faster without extensive tuning.
- The CrossEntropyLoss is chosen as it is the standard loss function for multi-class classification, comparing the model’s predicted class probabilities with the true labels to measure how wrong the predictions are.
- the accuracy function evaluates performance by selecting the class with the highest predicted score (using argmax) and calculating the proportion of correct predictions.
````python
# Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # For multi-class classification

def compute_accuracy(output,label):
  prediction = torch.argmax(output, dim=1)
  total_correct = (prediction == label).sum().item()
  accuracy = total_correct / len(label)
  return accuracy
````

