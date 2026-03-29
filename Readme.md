<img width="618" height="220" alt="image" src="https://github.com/user-attachments/assets/f26e44df-677a-49b2-bcb5-ec2f9053d295" />#  Neural Network Architecture (Multi Layered Perceptron)

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
<p align="center"><img width="911" height="155" alt="image" src="https://github.com/user-attachments/assets/4b0b3be3-c739-4178-85d3-ff4528e80490" /><p align="center">

## Training Neural Network 

This setup is designed to effectively train a multi-class classification model by combining a reliable optimizer, an appropriate loss function, and a simple evaluation metric.
- The Adam optimizer is used because it adaptively adjusts learning rates for each parameter, allowing the model to learn efficiently and converge faster without extensive tuning.
- The CrossEntropyLoss is chosen as it is the standard loss function for multi-class classification, comparing the model’s predicted class probabilities with the true labels to measure how wrong the predictions are.
- the accuracy function evaluates performance by selecting the class with the highest predicted score (using argmax) and calculating the proportion of correct predictions.
````python
# Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
````
````python
# Load Data with batch_size

batch_size = 128
epochs = 500

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # The training data is shuffled each epoch, which helps with model generalization.
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # The test data is not shuffled, ensuring consistent evaluation.

````
````python

losses = []             # List to store individual batch losses
epoch_losses = []       # Average loss per epoch for training
total_loss = []         # Total loss for training
train_accuracies = []   # Training accuracy per epoch

for epoch in range(epochs):
  model.train()       # Set the model to training model
  loss_count = 0      # Accumulate training loss for the current epoch
  total_correct=0     # Count correct prediction in training
  total_train = 0   # Total number of training sample produced

  for images, labels in train_dataloader:
    images, labels = images.to(device), labels.to(device)   # Move to GPU if available

    predicted_output = model(images)   # Forward Propagation to get predicted outcome
    loss = criterion(predicted_output, labels)   # Calculate the loss
    losses.append(loss.detach().cpu().numpy())      # Keep track of the loss

    optimizer.zero_grad()     # Reset gradient
    loss.backward()           # Backpropagation
    optimizer.step()          # Update weights

    # Track loss and accuracy
    loss_count += loss.item()     # Accumulate training loss
    _, predicted = torch.max(predicted_output, 1)
    total_train += labels.size(0)
    total_correct += (predicted == labels.argmax(dim=1)).sum().item()

  total_loss.append(loss_count)   # Calculate the total loss and save it to a variable called total_loss
  epoch_losses.append (loss_count/len(train_dataloader))  # Append the training loss
  train_accuracies.append(total_correct / total_train)    # Append the training accuracy

  print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_losses[-1]:.4f}, Total loss: {total_loss[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")

````
Initiate the model.eval() along with torch.no_grad() to turn off the gradients to evaluate model on test set after training

````python
model.eval()
correct = 0
total = 0

# Get the predictions for the test dataset
predicted_labels = []
true_labels = []

with torch.no_grad():
  for images, labels in test_dataloader:
    images, labels = images.to(device), labels.to(device) # Switch to GPU if available

    outputs = model(images)
    predicted = torch.max(outputs, 1)[1]  # Determine predicted classes
    total += labels.size(0)
    correct += (predicted == labels.argmax(dim=1)).sum().item()
    predicted_labels.extend(predicted.tolist())
    true_labels.extend(labels.argmax(dim=1).tolist())

accuracy = correct / total
print(f"Accuracy on the test set: {accuracy:.2%}")
````
Visualizing Training Results by plotting the learning curve:

````python
plt.plot(epoch_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.show()
````
<p align="center"><Figure size 640x480 with 1 Axes><img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/281d488e-dfde-4321-b826-17b7316c7cce" /><p align="center">

Plotting Confusion Matrix:

````python
# Import the packages for plotting the graph
import seaborn as sns
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(true_labels, predicted_labels)
# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
````
<p align="center"><Figure size 1000x800 with 1 Axes><img width="838" height="701" alt="image" src="https://github.com/user-attachments/assets/fa4a0075-7706-43e3-a56e-dbb23d547a4a" /><p align="center">

## Experiment 1:
In this architecture we will:

*  Add another fully connected layer
*  Increase hidden size to 512
*  Use batch normalization to stabalize learning process and speed up convergence

<p align="center"><p align="center"><img width="822" height="402" alt="image" src="https://github.com/user-attachments/assets/91ebf133-b598-45b2-80d6-b952b1257e4c" /><p align="center"><p align="center">

### Experiment Results: 
<p align="center"><img width="822" height="402" alt="image" src="https://github.com/user-attachments/assets/bd1d2e84-5299-45de-ac93-3b855e066504" /><p align="center">

<p align="center"><img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/c41effdd-2146-40a6-b065-6ab8c5ae013e" /><p align="center">

* The model obtains good results but severely overfits the training data as indicated by the divergence of training and validation loss and the relatively poorer performance on the testing set.
  
* Moving forward, experiment 2 will try to reduce overfitting by lowering the number of hidden neuron connections, adding batch_norm to layer 2, and adding L2 weight decay in Adam Optimizer.

## Experiment 2:
In this architecture we will

*   reduce number of neurons connection in hidden layer 2 from 512 -> 256
*   Add L2 weight decay to Adam optimizer
*   Add batch norm to layer 2
*   Depending on output of the training lower epochs

<p align="center"><img width="822" height="402" alt="image" src="https://github.com/user-attachments/assets/0733f875-7bb2-4e16-9287-3ae16df2e440" /><p align="center">

### Experiment Results: 
<p align="center"><img width="822" height="402" alt="image" src="https://github.com/user-attachments/assets/c1a53bfa-6e1c-44d9-9e56-29e82ddf44d3" /><p align="center">

<p align="center"><img width="575" height="455" alt="image" src="https://github.com/user-attachments/assets/10b3ccb6-8333-4f32-9e10-a6f1d2f29e27" /><p align="center">

1. Training Loss (blue) is relatively stable around ~0.29–0.31.
2. Validation Loss (orange) is consistently higher and fluctuates more (~0.31–0.36).
   
This could be due to insufficient regularization, a learning rate that’s slightly off, or a small validation set causing noisy estimates of loss. Since testing performance is also relatively poorer, it seems the model still does not generalize very well.

For the next experiment:
- Lower learning rate to help stabilise validation loss.
- Reduce the number of epochs to around 50, because evidence from the above graph indicates convergence occurring around those epochs
- Add another dropout layer in fc2 to regularize the second hidden layer

## Experiment 3:
In this experiment we will:
*   Lower learning rate to 0.001 or 0.0005
*   Reduce number of epoch to 50
*   Increase dropout rate 0.5 -> 0.6
*   Add another dropout layer in fc2 to regularize the second hidden layer
  
<p align="center"><img width="822" height="402" alt="image" src="https://github.com/user-attachments/assets/4c78735c-b2c3-4386-9223-94cb9accfaab"/><p align="center">

### Experiment Results: 
<p align="center"><img width="822" height="402" alt="image" src="https://github.com/user-attachments/assets/05dbcd36-8c73-463e-9ee1-3cadb267cb83" /><p align="center">

<p align="center"><img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/4b7b50b6-dc89-4126-9e1f-74a4a8c3a71f" /><p align="center">

After lowering Epoch, decreasing the learning rate, and adjusting the number of hidden connections and dropout rates for both layers 1 and 2, we achieve a more stable version with minimal overfitting, while maintaining the target of over 80% accuracy.

### Confusion Matrix
<p align="center"><img width="838" height="701" alt="image" src="https://github.com/user-attachments/assets/4c2d5a3b-5d4a-405d-9809-8ca6b00bd909" /><p align="center">
The model seems to struggle to distinguish between the numbers 2 & 5, 4 & 7, and 0 & 4 the most.

## Limitations
While we achieved the target of 80% accuracy there is still room for improvement. There is still a sign of slight overfit meaning the generalizability of the model could worsen over time. Other different combinations of the number of hidden connections and dropout rate could potentially be explored. More hidden layers could also be added with different dropout rates to find the best combinations that yield the most optimal performance/overfit tradeoff.






