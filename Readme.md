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

#2. Download Dataset
# Mount Google Drive
drive.mount('/content/gdrive')

# Create a working folder & set working directory
%cd '/content/gdrive/MyDrive/DL_ASG_1'



````
