"""
PyTorch Deep Learning Tutorial for Beginners - FashionMNIST Image Classification

This script demonstrates how to build, train, and evaluate a simple neural network
using PyTorch for the FashionMNIST dataset image classification task.

Learning Objectives:
1. Understand basic PyTorch concepts and syntax
2. Learn how to load and preprocess data
3. Master neural network definition and construction
4. Understand training loops and optimization process
5. Learn how to evaluate model performance
"""


# Import Required Libraries
# -------------------------
import torch                    # PyTorch core library for tensor operations and autograd
# Neural network module containing various layers and activation functions
from torch import nn
from torch.utils.data import DataLoader  # Data loader for batch processing
from torchvision import datasets          # Predefined datasets
# Transform to convert images to tensors
from torchvision.transforms import ToTensor
from tqdm import tqdm          # Progress bar library for displaying training progress


# Step 1: Dataset Preparation
# ---------------------------
print("Preparing FashionMNIST dataset...")

# Load FashionMNIST training dataset
# FashionMNIST is a classic dataset containing 10 classes of clothing images
# Each image is a 28x28 pixel grayscale image
training_data = datasets.FashionMNIST(
    root="data",               # Root directory for data storage
    train=True,                # Specify this is training dataset
    download=True,             # Automatically download if data doesn't exist
    transform=ToTensor(),      # Convert images to PyTorch tensor format
)

# Load FashionMNIST test dataset
# Test data is used to evaluate model performance on unseen data
test_data = datasets.FashionMNIST(
    root="data",               # Root directory for data storage
    train=False,               # Specify this is test dataset
    download=True,             # Automatically download if data doesn't exist
    transform=ToTensor(),      # Convert images to PyTorch tensor format
)

# Set batch size
# Batch size determines how many images are processed at once during training
# Larger batch sizes can speed up training but require more memory
batch_size = 64

# Create data loaders
# DataLoader wraps the dataset into an iterable object supporting batch processing, shuffling, etc.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Display sample batch dimensions
# This helps understand the shape and structure of the data
print("\nDataset Information:")
for X, y in test_dataloader:
    print("Image tensor X shape [N, C, H, W]: ", X.shape)
    print("   - N: Batch size")
    print("   - C: Number of channels (FashionMNIST is grayscale, so C=1)")
    print("   - H: Image height (28 pixels)")
    print("   - W: Image width (28 pixels)")
    print("Label tensor y shape: ", y.shape)
    print("   - Each label is a number between 0-9, representing different clothing categories")
    break  # Only show first batch to avoid printing too much information


# Step 2: Device Configuration
# ----------------------------
# Check if GPU is available, use GPU if available, otherwise use CPU
# GPU can significantly accelerate deep learning model training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")


# Step 3: Define Neural Network Architecture
# ------------------------------------------
class NeuralNetwork(nn.Module):
    """
    Simple fully connected neural network class

    This network contains:
    - A flatten layer: converts 28x28 images to 784-dimensional vectors
    - Two hidden layers: each with 512 neurons, using ReLU activation function
    - An output layer: outputs 10 values corresponding to 10 clothing categories

    Inherits from nn.Module, which is the base class for all neural networks in PyTorch
    """

    def __init__(self):
        """
        Initialize network structure
        Define various layers and parameters of the network
        """
        super().__init__()  # Call parent class initialization method

        # Flatten layer: converts 2D images (28x28) to 1D vectors (784)
        # This is necessary because fully connected layers can only process 1D input
        self.flatten = nn.Flatten()

        # Use Sequential container to combine multiple layers
        # Data will flow through these layers in sequence
        self.linear_relu_stack = nn.Sequential(
            # First fully connected layer: 784 input -> 512 output
            nn.Linear(28 * 28, 512),
            # ReLU activation function: adds non-linearity, helps network learn complex patterns
            nn.ReLU(),
            # Second fully connected layer: 512 input -> 512 output
            nn.Linear(512, 512),
            # Second ReLU activation function
            nn.ReLU(),
            # Output layer: 512 input -> 10 output (corresponding to 10 categories)
            # Note: No activation function here because CrossEntropyLoss will be used later
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """
        Forward propagation function

        Defines how data flows through the network
        Args:
            x: Input tensor with shape [batch_size, 1, 28, 28]
        Returns:
            logits: Output tensor with shape [batch_size, 10]
        """
        # Flatten the input image
        x = self.flatten(x)
        # Pass through fully connected layers and activation functions
        logits = self.linear_relu_stack(x)
        return logits


# Create neural network instance and move it to specified device (GPU or CPU)
model = NeuralNetwork().to(device)
print(f"\nNeural Network Structure:")
print(model)

# Calculate total number of model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")


# Step 4: Training Configuration
# ------------------------------
print(f"\nTraining Configuration:")

# Loss function: Cross-entropy loss
# Cross-entropy loss is the standard loss function for classification problems
# It measures the difference between predicted probability distribution and true labels
loss_fn = nn.CrossEntropyLoss()
print(f"Loss function: {loss_fn.__class__.__name__}")

# Optimizer: Stochastic Gradient Descent (SGD)
# Optimizer is responsible for updating network parameters to minimize loss function
# Learning rate (lr) controls the step size of each parameter update
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Learning rate: {1e-3}")

# Number of training epochs
# One epoch means the model has seen the complete training data once
epochs = 30
print(f"Number of epochs: {epochs}")


# Step 5: Define Training and Testing Functions
# ---------------------------------------------
def train(dataloader, model, loss_fn, optimizer):
    """
    Train the model for one complete epoch

    This function performs the following steps:
    1. Set model to training mode
    2. Iterate through training data
    3. Forward propagation to compute predictions
    4. Compute loss
    5. Backward propagation to compute gradients
    6. Update model parameters

    Args:
        dataloader: Training data loader
        model: Neural network model
        loss_fn: Loss function
        optimizer: Optimizer
    """
    # Set model to training mode
    # This enables training-specific features like dropout, batch normalization, etc.
    model.train()

    # Record loss values for each batch to monitor training process
    loss_record = []

    # Create progress bar to display training progress
    # tqdm makes training process more intuitive, showing real-time progress and loss values
    train_pbar = tqdm(dataloader, position=0, leave=True)

    # Iterate through each batch in training data
    for image, label in train_pbar:
        # Zero the gradients
        # This is necessary because PyTorch accumulates gradients
        # If not zeroed, gradients will accumulate, causing unstable training
        optimizer.zero_grad()

        # Move data to specified device (GPU or CPU)
        image, label = image.to(device), label.to(device)

        # Forward propagation: compute model predictions
        pred = model(image)

        # Compute loss: compare predictions with true labels
        loss = loss_fn(pred, label)

        # Backward propagation: compute gradients
        # This is the core of automatic differentiation, PyTorch automatically computes gradients for all parameters
        loss.backward()

        # Update parameters: use computed gradients to update network weights
        optimizer.step()

        # Record loss value (convert to Python value to avoid memory leaks)
        loss_record.append(loss.detach().item())

        # Update progress bar display
        # Show current epoch and loss value
        train_pbar.set_description(f'Epoch [{epoch+1}/{epochs}]')
        train_pbar.set_postfix({'loss': loss.detach().item()})


def test(dataloader, model, loss_fn):
    """
    Evaluate model performance on test set

    This function computes model accuracy and average loss on unseen data
    This is an important metric for evaluating model generalization ability

    Args:
        dataloader: Test data loader
        model: Trained neural network model
        loss_fn: Loss function
    """
    # Get total size and number of batches in test set
    size = len(dataloader.dataset)      # Total number of test samples
    num_batches = len(dataloader)       # Number of test batches

    # Set model to evaluation mode
    # This disables training-specific features like dropout
    model.eval()

    # Initialize test loss and correct prediction count
    test_loss, correct = 0, 0

    # Don't compute gradients during testing to save memory and computation time
    with torch.no_grad():
        # Iterate through each batch in test data
        for X, y in dataloader:
            # Move data to specified device
            X, y = X.to(device), y.to(device)

            # Forward propagation: compute model predictions
            pred = model(X)

            # Accumulate batch loss
            test_loss += loss_fn(pred, y).item()

            # Count correct predictions
            # pred.argmax(1) returns the index of maximum value in each row (i.e., predicted class)
            # Compare with true labels y to count correct predictions
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Compute average loss and accuracy
    test_loss /= num_batches           # Average loss
    correct /= size                    # Accuracy (between 0-1)

    # Print test results
    print(
        f"Test Results:\n Accuracy: {(100*correct):>0.1f}%,"
        + f" Average loss: {test_loss:>8f}\n")


# Step 6: Main Training Loop
# --------------------------
print("Starting training!")
print("=" * 50)

# Main training loop: repeat training and testing
for epoch in range(epochs):
    print(f"\n[Epoch {epoch + 1}]")
    print("-" * 30)

    # Train model for one epoch
    train(train_dataloader, model, loss_fn, optimizer)

    # Evaluate model performance on test set
    test(test_dataloader, model, loss_fn)

print("=" * 50)
print("[Training completed!]")


# Learning Summary
# ----------------
print(f"\n🎉 Congratulations! You have successfully trained a neural network!")
print(f"\n📚 What you learned:")
print(f"1. How to load and preprocess data using PyTorch")
print(f"2. How to define neural network architecture")
print(f"3. How to set up loss functions and optimizers")
print(f"4. How to implement training loops")
print(f"5. How to evaluate model performance")
print(f"\n🔍 Next steps suggestions:")
print(f"1. Try adjusting network structure (add more layers, change neuron counts)")
print(f"2. Experiment with different optimizers and learning rates")
print(f"3. Add regularization techniques (like Dropout)")
print(f"4. Try other datasets (like CIFAR-10, MNIST)")
print(f"5. Learn more advanced architectures (like CNN, RNN)")
