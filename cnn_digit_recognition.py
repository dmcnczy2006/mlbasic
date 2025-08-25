"""
PyTorch Deep Learning Tutorial for Beginners - MNIST Handwritten Digit Recognition

This script demonstrates how to build, train, and evaluate a neural network
using PyTorch for the MNIST dataset handwritten digit classification task.

Learning Objectives:
1. Understand how to work with image classification problems
2. Learn about convolutional neural networks (CNNs)
3. Master data preprocessing and augmentation techniques
4. Understand model evaluation and visualization
5. Learn how to save and load trained models
"""


# Import Required Libraries
# -------------------------
import torch                    # PyTorch core library for tensor operations and autograd
# Neural network module containing various layers and activation functions
import torch.nn as nn
import torch.optim as optim    # Optimization algorithms
from torch.utils.data import DataLoader  # Data loader for batch processing
# Predefined datasets and image transformations
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # For plotting and visualization
import numpy as np             # For numerical operations
from tqdm import tqdm          # Progress bar library for displaying training progress


# Step 1: Dataset Preparation and Data Augmentation
# -------------------------------------------------
print("Preparing MNIST dataset with data augmentation...")

# Define data transformations
# Data augmentation helps improve model generalization by creating variations of training data
transform_train = transforms.Compose([
    # Randomly rotate images by ±10 degrees
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(
        0.1, 0.1)),  # Random translation
    transforms.ToTensor(),                # Convert images to PyTorch tensor format
    # Normalize with MNIST mean and std
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),                # Convert images to PyTorch tensor format
    # Normalize with MNIST mean and std
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST training dataset
# MNIST contains 60,000 training images of handwritten digits (0-9)
# Each image is 28x28 pixels in grayscale
training_data = datasets.MNIST(
    root="data",               # Root directory for data storage
    train=True,                # Specify this is training dataset
    download=True,             # Automatically download if data doesn't exist
    transform=transform_train,  # Apply training transformations
)

# Load MNIST test dataset
# Test set contains 10,000 images for evaluation
test_data = datasets.MNIST(
    root="data",               # Root directory for data storage
    train=False,               # Specify this is test dataset
    download=True,             # Automatically download if data doesn't exist
    transform=transform_test,  # Apply test transformations
)

# Set batch size
# Batch size determines how many images are processed at once during training
# Larger batch sizes can speed up training but require more memory
batch_size = 128

# Create data loaders
# DataLoader wraps the dataset into an iterable object supporting batch processing, shuffling, etc.
train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Display sample batch dimensions and visualize some images
print("\nDataset Information:")
for X, y in test_dataloader:
    print("Image tensor X shape [N, C, H, W]: ", X.shape)
    print("   - N: Batch size")
    print("   - C: Number of channels (MNIST is grayscale, so C=1)")
    print("   - H: Image height (28 pixels)")
    print("   - W: Image width (28 pixels)")
    print("Label tensor y shape: ", y.shape)
    print("   - Each label is a number between 0-9, representing handwritten digits")
    break

# Visualize some sample images


def visualize_samples(dataloader, num_samples=8):
    """Display sample images from the dataset"""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i, (images, labels) in enumerate(dataloader):
        if i >= num_samples:
            break
        # Denormalize the image for display
        img = images[i].squeeze().numpy()
        img = (img * 0.3081) + 0.1307  # Reverse normalization
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Digit: {labels[i].item()}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


print("\nVisualizing sample images...")
# visualize_samples(test_dataloader)


# Step 2: Device Configuration
# ----------------------------
# Check if GPU is available, use GPU if available, otherwise use CPU
# GPU can significantly accelerate deep learning model training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")


# Step 3: Define Convolutional Neural Network Architecture
# -------------------------------------------------------
class ConvNet(nn.Module):
    """
    Convolutional Neural Network for digit recognition

    This network contains:
    - Two convolutional layers with ReLU activation and max pooling
    - Two fully connected layers with dropout for regularization
    - Output layer for 10 digit classes (0-9)

    CNNs are excellent for image processing because they can learn spatial features
    like edges, textures, and patterns at different scales.
    """

    def __init__(self):
        """
        Initialize network structure
        Define various layers and parameters of the network
        """
        super().__init__()  # Call parent class initialization method

        # First convolutional block
        # Conv2d: 2D convolutional layer
        # Input: 1 channel (grayscale), Output: 32 feature maps, Kernel: 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()  # ReLU activation function
        # Max pooling: reduces spatial dimensions by half
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolutional block
        # Input: 32 channels, Output: 64 feature maps, Kernel: 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Calculate the size after convolutions and pooling
        # Input: 28x28 -> After pool1: 14x14 -> After pool2: 7x7
        # With 64 channels: 7 * 7 * 64 = 3136 features

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 128)  # 3136 -> 128
        self.relu3 = nn.ReLU()
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)  # 128 -> 10 (output classes)

    def forward(self, x):
        """
        Forward propagation function

        Defines how data flows through the network
        Args:
            x: Input tensor with shape [batch_size, 1, 28, 28]
        Returns:
            logits: Output tensor with shape [batch_size, 10]
        """
        # First convolutional block
        x = self.pool1(self.relu1(self.conv1(x)))

        # Second convolutional block
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten the output for fully connected layers
        # From [batch_size, 64, 7, 7] to [batch_size, 3136]
        x = x.view(-1, 7 * 7 * 64)

        # Fully connected layers
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)

        return x


# Create neural network instance and move it to specified device (GPU or CPU)
model = ConvNet().to(device)
print(f"\nConvolutional Neural Network Structure:")
print(model)

# Calculate total number of model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
print(f"Total model parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# Step 4: Training Configuration
# ------------------------------
print(f"\nTraining Configuration:")

# Loss function: Cross-entropy loss
# Cross-entropy loss is the standard loss function for classification problems
# It measures the difference between predicted probability distribution and true labels
loss_fn = nn.CrossEntropyLoss()
print(f"Loss function: {loss_fn.__class__.__name__}")

# Optimizer: Adam optimizer
# Adam is an adaptive learning rate optimizer that often works better than SGD
# It automatically adjusts learning rates for each parameter
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Learning rate: {0.001}")

# Learning rate scheduler
# Reduces learning rate when training plateaus, helping convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Number of training epochs
# One epoch means the model has seen the complete training data once
epochs = 20
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
    Returns:
        average_loss: Average loss for this epoch
    """
    # Set model to training mode
    # This enables training-specific features like dropout, batch normalization, etc.
    model.train()

    # Record loss values for each batch to monitor training process
    running_loss = 0.0
    num_batches = len(dataloader)

    # Create progress bar to display training progress
    # tqdm makes training process more intuitive, showing real-time progress and loss values
    train_pbar = tqdm(dataloader, position=0, leave=True)

    # Iterate through each batch in training data
    for batch_idx, (images, labels) in enumerate(train_pbar):
        # Move data to specified device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        # This is necessary because PyTorch accumulates gradients
        # If not zeroed, gradients will accumulate, causing unstable training
        optimizer.zero_grad()

        # Forward propagation: compute model predictions
        outputs = model(images)

        # Compute loss: compare predictions with true labels
        loss = loss_fn(outputs, labels)

        # Backward propagation: compute gradients
        # This is the core of automatic differentiation, PyTorch automatically computes gradients for all parameters
        loss.backward()

        # Update parameters: use computed gradients to update network weights
        optimizer.step()

        # Accumulate loss for this epoch
        running_loss += loss.item()

        # Update progress bar display
        # Show current batch and loss value
        train_pbar.set_description(f'Training')
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate average loss for this epoch
    average_loss = running_loss / num_batches
    return average_loss


def test(dataloader, model, loss_fn):
    """
    Evaluate model performance on test set

    This function computes model accuracy and average loss on unseen data
    This is an important metric for evaluating model generalization ability

    Args:
        dataloader: Test data loader
        model: Trained neural network model
        loss_fn: Loss function
    Returns:
        average_loss: Average loss on test set
        accuracy: Accuracy on test set
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
        for images, labels in dataloader:
            # Move data to specified device
            images, labels = images.to(device), labels.to(device)

            # Forward propagation: compute model predictions
            outputs = model(images)

            # Accumulate batch loss
            test_loss += loss_fn(outputs, labels).item()

            # Count correct predictions
            # outputs.argmax(1) returns the index of maximum value in each row (i.e., predicted class)
            # Compare with true labels labels to count correct predictions
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    # Compute average loss and accuracy
    average_loss = test_loss / num_batches           # Average loss
    accuracy = correct / size                        # Accuracy (between 0-1)

    # Print test results
    print(
        f"Test Results:\n Accuracy: {(100*accuracy):>0.1f}%,"
        + f" Average loss: {average_loss:>8f}")

    return average_loss, accuracy


# Step 6: Main Training Loop
# --------------------------
print("Starting training!")
print("=" * 60)

# Lists to store training history for plotting
train_losses = []
test_losses = []
test_accuracies = []

# Main training loop: repeat training and testing
for epoch in range(epochs):
    print(f"\n[Epoch {epoch + 1}/{epochs}]")
    print("-" * 40)

    # Train model for one epoch
    train_loss = train(train_dataloader, model, loss_fn, optimizer)

    # Evaluate model performance on test set
    test_loss, test_accuracy = test(test_dataloader, model, loss_fn)

    # Store metrics for plotting
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # Update learning rate based on test loss
    scheduler.step(test_loss)

    print(f"Training loss: {train_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {100*test_accuracy:.2f}%")

print("=" * 60)
print("[Training completed!]")


# Step 7: Training Visualization and Analysis
# -------------------------------------------
print("\nGenerating training plots...")

# Create subplots for loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot training and test loss
ax1.plot(train_losses, label='Training Loss', color='blue')
ax1.plot(test_losses, label='Test Loss', color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss Over Time')
ax1.legend()
ax1.grid(True)

# Plot test accuracy
ax2.plot(test_accuracies, label='Test Accuracy', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Test Accuracy Over Time')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print final results
final_accuracy = test_accuracies[-1]
print(f"\n🎯 Final Test Accuracy: {100*final_accuracy:.2f}%")

if final_accuracy > 0.95:
    print("🌟 Excellent performance! Your model is working very well!")
elif final_accuracy > 0.90:
    print("👍 Good performance! Your model is learning effectively.")
elif final_accuracy > 0.80:
    print("✅ Decent performance! There's room for improvement.")
else:
    print("📚 Keep learning! Try adjusting hyperparameters or architecture.")


# Step 8: Model Evaluation and Visualization
# -----------------------------------------
print("\nEvaluating model on sample images...")


def visualize_predictions(dataloader, model, num_samples=12):
    """Display sample images with their predictions"""
    model.eval()

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.ravel()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_samples:
                break

            # Get prediction
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Display image
            img = images[i].squeeze().cpu().numpy()
            img = (img * 0.3081) + 0.1307  # Reverse normalization

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(
                f'True: {labels[i].item()}\nPredicted: {predicted[i].item()}')

            # Color code: green for correct, red for incorrect
            if predicted[i].item() == labels[i].item():
                axes[i].set_title(
                    f'True: {labels[i].item()}\nPredicted: {predicted[i].item()}\n✅', color='green')
            else:
                axes[i].set_title(
                    f'True: {labels[i].item()}\nPredicted: {predicted[i].item()}\n❌', color='red')

            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Visualize predictions
visualize_predictions(test_dataloader, model)


# Step 9: Save the Trained Model
# -------------------------------
print("\nSaving the trained model...")

# Save the entire model
torch.save(model, 'mnist_digit_recognition_model.pth')
print("Model saved as 'mnist_digit_recognition_model.pth'")

# Save just the model parameters (state dict)
torch.save(model.state_dict(), 'mnist_digit_recognition_state_dict.pth')
print("Model state dict saved as 'mnist_digit_recognition_state_dict.pth'")

# Demonstrate how to load the model
print("\nDemonstrating model loading...")
loaded_model = torch.load('mnist_digit_recognition_model.pth')
loaded_model.eval()
print("Model loaded successfully!")


# Learning Summary
# ----------------
print(f"\n🎉 Congratulations! You have successfully trained a CNN for digit recognition!")
print(f"\n📚 What you learned:")
print(f"1. How to work with image datasets and data augmentation")
print(f"2. How to design and implement convolutional neural networks")
print(f"3. How to use advanced optimizers and learning rate scheduling")
print(f"4. How to visualize training progress and model predictions")
print(f"5. How to save and load trained models")
print(f"\n🔍 Next steps suggestions:")
print(f"1. Try different CNN architectures (ResNet, VGG, etc.)")
print(f"2. Experiment with more data augmentation techniques")
print(f"3. Implement ensemble methods for better accuracy")
print(f"4. Try transfer learning with pre-trained models")
print(f"5. Apply to other image classification tasks (CIFAR-10, etc.)")
print(f"6. Learn about attention mechanisms and modern architectures")
print(f"7. Explore deployment options (ONNX, TorchScript, etc.)")

print(f"\n🚀 Your journey in deep learning has just begun!")
print(f"Keep experimenting, keep learning, and most importantly, have fun!")
