=========================================================
Exercise 4B: PyTorch Framework Deep Dive
=========================================================

Topics Covered
1. Neural Networks using torch.nn.Module
2. CNN using Conv2d and MaxPool2d
3. DataLoader
4. Dropout Regularization
5. Weight Decay Regularization
6. Learning Rate Scheduling
7. GPU Support
8. Model Saving and Loading

Dataset: MNIST
=========================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# =====================================================
# Device Configuration
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("PyTorch Framework Deep Dive")
print("=" * 60)

print("Device:", device)

# =====================================================
# Hyperparameters
# =====================================================

BATCH_SIZE = 64

LEARNING_RATE = 0.001

EPOCHS = 5

WEIGHT_DECAY = 1e-4

# =====================================================
# Data Transformation
# =====================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    )
])

# =====================================================
# Download Dataset
# =====================================================

print("\nDownloading MNIST Dataset...\n")

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# =====================================================
# Data Loader
# =====================================================

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(
    "Training Samples:",
    len(train_dataset)
)

print(
    "Testing Samples:",
    len(test_dataset)
)

# =====================================================
# CNN MODEL
# =====================================================

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        # -----------------------------------------
        # First Convolution Layer
        # -----------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # -----------------------------------------
        # Second Convolution Layer
        # -----------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # -----------------------------------------
        # Fully Connected Layers
        # -----------------------------------------

        self.fc1 = nn.Linear(
            32 * 7 * 7,
            128
        )

        # Dropout Regularization

        self.dropout = nn.Dropout(
            p=0.5
        )

        self.fc2 = nn.Linear(
            128,
            10
        )

        self.relu = nn.ReLU()

    # =============================================
    # Forward Pass
    # =============================================

    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.pool2(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x

# =====================================================
# Model Creation
# =====================================================

model = CNN().to(device)

print("\nModel Architecture\n")
print(model)
# =====================================================
# Loss Function
# =====================================================

criterion = nn.CrossEntropyLoss()

# =====================================================
# Optimizer
# Weight Decay = L2 Regularization
# =====================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# =====================================================
# Learning Rate Scheduler
# =====================================================

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.5
)

# =====================================================
# Training Function
# =====================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        # -------------------------------
        # Clear Gradients
        # -------------------------------

        optimizer.zero_grad()

        # -------------------------------
        # Forward Pass
        # -------------------------------

        outputs = model(images)

        # -------------------------------
        # Compute Loss
        # -------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # -------------------------------
        # Backpropagation
        # -------------------------------

        loss.backward()

        # -------------------------------
        # Update Parameters
        # -------------------------------

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    accuracy = (
        100 * correct / total
    )

    return avg_loss, accuracy

# =====================================================
# Evaluation Function
# =====================================================

def evaluate():

    model.eval()

    correct = 0

    total = 0

    running_loss = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    avg_loss = (
        running_loss /
        len(test_loader)
    )

    accuracy = (
        100 * correct / total
    )

    return avg_loss, accuracy

# =====================================================
# Lists for Plotting
# =====================================================

train_losses = []

test_losses = []

train_accuracies = []

test_accuracies = []

# =====================================================
# Training Loop
# =====================================================

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch()

    test_loss, test_acc = evaluate()

    current_lr = optimizer.param_groups[0]["lr"]

    train_losses.append(
        train_loss
    )

    test_losses.append(
        test_loss
    )

    train_accuracies.append(
        train_acc
    )

    test_accuracies.append(
        test_acc
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}]"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : {train_acc:.2f}%"
    )

    print(
        f"Test Loss  : {test_loss:.4f}"
    )

    print(
        f"Test Acc   : {test_acc:.2f}%"
    )

    print(
        f"Learning Rate : {current_lr:.6f}"
    )

    print("-" * 50)

    # --------------------------------
    # Scheduler Step
    # --------------------------------

    scheduler.step()
  # =====================================================
# Final Evaluation
# =====================================================

final_test_loss, final_test_accuracy = evaluate()

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Final Test Loss     : "
    f"{final_test_loss:.4f}"
)

print(
    f"Final Test Accuracy : "
    f"{final_test_accuracy:.2f}%"
)

# =====================================================
# Save Model
# =====================================================

MODEL_PATH = "cnn_mnist_model.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(
    f"\nModel saved as: "
    f"{MODEL_PATH}"
)

# =====================================================
# Load Model
# =====================================================

loaded_model = CNN().to(device)

loaded_model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

loaded_model.eval()

print(
    "Model loaded successfully."
)

# =====================================================
# Verify Loaded Model
# =====================================================

correct = 0

total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = loaded_model(images)

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

loaded_accuracy = (
    100 * correct / total
)

print(
    f"Loaded Model Accuracy: "
    f"{loaded_accuracy:.2f}%"
)

# =====================================================
# Plot Learning Curves
# =====================================================

plt.figure(
    figsize=(12, 5)
)

# -----------------------------------------
# Loss Curve
# -----------------------------------------

plt.subplot(
    1,
    2,
    1
)

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    test_losses,
    label="Test Loss"
)

plt.title(
    "Loss Curve"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(True)

# -----------------------------------------
# Accuracy Curve
# -----------------------------------------

plt.subplot(
    1,
    2,
    2
)

plt.plot(
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    test_accuracies,
    label="Test Accuracy"
)

plt.title(
    "Accuracy Curve"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# =====================================================
# Display Sample Predictions
# =====================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

loaded_model.eval()

images, labels = next(
    iter(test_loader)
)

images = images.to(device)

labels = labels.to(device)

with torch.no_grad():

    outputs = loaded_model(images)

    _, predictions = torch.max(
        outputs,
        dim=1
    )

for i in range(10):

    print(
        f"Image {i+1:2d} | "
        f"Actual: {labels[i].item()} | "
        f"Predicted: {predictions[i].item()}"
    )

# =====================================================
# Summary of Exercise Requirements
# =====================================================

print("\n" + "=" * 60)
print("EXERCISE 4B REQUIREMENTS COVERED")
print("=" * 60)

print(
    "1. torch.nn.Module               ✓"
)

print(
    "2. Conv2d and MaxPool2d          ✓"
)

print(
    "3. DataLoader                    ✓"
)

print(
    "4. Dropout Regularization        ✓"
)

print(
    "5. Weight Decay Regularization   ✓"
)

print(
    "6. Learning Rate Scheduler       ✓"
)

print(
    "7. GPU Support                   ✓"
)

print(
    "8. Model Saving / Loading        ✓"
)

print(
    "9. Accuracy Evaluation           ✓"
)

print(
    "10. Learning Curves              ✓"
)

print("\nExercise 4B Completed Successfully!")
