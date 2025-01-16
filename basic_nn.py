"""
basic_nn.py

PURPOSE:
  - Train and compare a simple (but slightly deeper) fully-connected neural network (MLP)
    to classify facial expressions into 7 categories (e.g., angry, happy, etc.).
  - Serves as a middle ground between the simplest models (baseline, softmax) and
    advanced convolutional networks (CNN).
  - Illustrates how a basic MLP can learn more effectively than a linear model,
    yet still not exploit spatial (2D) structure like a CNN.

FUNCTIONS & FLOW:
  1) Main steps:
     - Load processed train/val data (train_data.pt, val_data.pt).
     - Build an MLP (BasicNN) with two hidden layers, ReLU, BatchNorm, and Dropout.
     - Train for N epochs, tracking train/val loss each epoch.
     - Print final classification report and accuracy for validation set.
     - evaluate on a test set .
     - Plot train vs val loss across epochs.

  2) Model Explanation:
     - Input: Flattened 48x48 grayscale images => vector of size 2304.
     - Two hidden layers (dim1=128, dim2=64) each with BN, ReLU, Dropout(0.3).
     - Output layer => 7 classes with CrossEntropyLoss.

  3) This code helps us compare:
     - baseline.py (always picks the majority class),
     - softmax.py (single linear + softmax),
     - basic_nn.py (this script, an MLP),
     - advanced_network.py (CNN).

HOW TO RUN:
  python basic_nn.py

REQUIREMENTS:
  - Torch, NumPy, sklearn, Matplotlib
  - Processed .pt files in 'processed_data/' (train_data.pt, val_data.pt, optional test_data.pt)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader

###############################################################################
# 1) Define the Basic Neural Network (MLP)
###############################################################################
class BasicNN(nn.Module):
    """
    A simple MLP with two hidden layers. Uses:
     - Linear layer, BatchNorm, ReLU, Dropout in each hidden layer
     - Final layer -> 'num_classes' outputs
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):
        super(BasicNN, self).__init__()
        # First fully-connected layer: from 'input_dim' to 'hidden_dim1'
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)  # BN helps stabilize training
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)         # 0.3 dropout fraction

        # Second fully-connected layer: from hidden_dim1 to hidden_dim2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        # Final output layer: from hidden_dim2 to num_classes
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        """
        Forward pass:
          1) fc1 -> BN1 -> ReLU -> Dropout
          2) fc2 -> BN2 -> ReLU -> Dropout
          3) fc3 -> raw logits for each class
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x

###############################################################################
# 2) Helper Functions
###############################################################################
def flatten_images(images):
    """
    Reshape 4D tensor of shape (N,1,48,48) into (N, 2304).
    Because an MLP doesn't handle 2D structure, we flatten the 48x48 pixels.
    """
    N = images.shape[0]
    return images.view(N, -1)

def train_model(model, loader, criterion, optimizer):
    """
    Performs one epoch of training on 'model' with the given dataloader 'loader'.
    Returns the average train loss across the entire dataset.
    Steps:
      1) model.train() to enable dropout, BN in training mode
      2) zero_grad -> forward -> compute loss -> backward -> step
      3) accumulate total loss for reporting
    """
    model.train()
    total_loss = 0
    for data, targets in loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)  # multiply by batch_size
    return total_loss / len(loader.dataset)

def validate_model(model, loader, criterion):
    """
    Evaluates the model on a validation (or test) set.
    Returns:
      - val_loss: average loss over the entire loader
      - val_preds: all predicted labels
      - val_targets: all ground-truth labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * data.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = total_loss / len(loader.dataset)
    val_preds = np.concatenate(all_preds)
    val_targets = np.concatenate(all_targets)
    return val_loss, val_preds, val_targets

###############################################################################
# 3) Main Execution
###############################################################################
def main():
    """
    MAIN function:
      1. Print banner for "BASIC FULLY CONNECTED NEURAL NETWORK (MLP) FOR EMOTIONS".
      2. Load train_data.pt and val_data.pt from 'processed_data/'.
      3. Flatten images and build DataLoaders.
      4. Construct BasicNN with two hidden layers, BN, Dropout.
      5. Train for 'num_epochs'. Keep track of train & val losses.
      6. Print classification report + final accuracy on validation set.
      7. Save results to a text file, plot the training curve.
      8.  Evaluate on test set.
    """
    print("=========================================================")
    print("  BASIC FULLY CONNECTED NEURAL NETWORK (MLP) FOR EMOTIONS ")
    print("=========================================================\n")

    # 1) Make sure we have a 'results' folder
    os.makedirs("results", exist_ok=True)

    # 2) Paths to .pt files
    train_path = "processed_data/train_data.pt"
    val_path   = "processed_data/val_data.pt"
    test_path  = "processed_data/test_data.pt"

    # Basic check for existence
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("[Error] Processed train/val data not found.")
        return

    # 3) Load the train/val .pt files
    # weights_only=False means we can load the pickled tuple (images, labels, dict)
    train_images, train_labels, label_to_idx = torch.load(train_path, weights_only=False)
    val_images,   val_labels,   _            = torch.load(val_path, weights_only=False)

    print("[Info] Train set size:", train_labels.size(0))
    print("[Info] Val set size:  ", val_labels.size(0), "\n")

    # Flatten the images from (N,1,48,48) => (N,2304)
    train_images_flat = flatten_images(train_images)
    val_images_flat   = flatten_images(val_images)

    # Build Datasets & DataLoaders
    train_dataset = TensorDataset(train_images_flat, train_labels)
    val_dataset   = TensorDataset(val_images_flat, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 4) Create the BasicNN model
    input_dim   = train_images_flat.shape[1]    # e.g. 2304
    hidden_dim1 = 128
    hidden_dim2 = 64
    num_classes = len(label_to_idx)

    # Model instantiation
    model = BasicNN(input_dim, hidden_dim1, hidden_dim2, num_classes)

    # Set up criterion & optimizer
    criterion = nn.CrossEntropyLoss()  # default cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5) Training loop
    num_epochs = 20
    train_losses = []
    val_losses   = []

    print("[Info] Starting training...\n")
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_preds, val_targets = validate_model(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 6) Evaluate classification metrics on validation
    # We used 'val_preds' and 'val_targets' from last validation step
    # classification_report requires string or list of target names
    # label_to_idx is a dict: {"angry":0, "disgust":1, ...}. We can do label_to_idx.keys().
    # zero_division=0 to avoid warnings on classes with no predictions
    report = classification_report(val_targets, val_preds, target_names=label_to_idx.keys(), zero_division=0)
    val_accuracy = accuracy_score(val_targets, val_preds)

    # Print classification report
    print("\n===============================")
    print(" Validation Classification Report:")
    print("===============================")
    print(report)
    print(f"[Summary] Final Validation Accuracy: {val_accuracy * 100:.2f}%\n")

    # Save results to file
    results_file = "results/basic_nn_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Final Accuracy: {val_accuracy:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"[Info] Results saved to '{results_file}' for comparison.\n")

    # Plot training vs validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses,   label='Val Loss',   marker='s')
    plt.title('Basic NN (Deeper) Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n[Info] Basic NN training completed.\n")

    # 7) Additional summary: accuracy on train, val, and test
    print("------------------------------------------------------------")
    print("ADDITIONAL SUMMARY: BASIC NN EVALUATION ON TRAIN, VAL, TEST")
    print("------------------------------------------------------------\n")

    def get_preds_metrics(model, data_x, data_y):
        """
        Utility to compute model predictions on a given dataset,
        then measure accuracy, precision, recall, etc.
        Returns (acc, prec, rec).
        """
        model.eval()
        with torch.no_grad():
            outputs = model(data_x)
            preds = torch.argmax(outputs, dim=1)
        y_true = data_y.numpy()
        y_pred = preds.numpy()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
        return acc, prec, rec

    # Evaluate on the entire train set
    train_acc, _, _ = get_preds_metrics(model, train_images_flat, train_labels)
    print(f"Train Set -> Accuracy: {train_acc * 100:.2f}%")

    # Evaluate on the entire val set
    val_acc, _, _ = get_preds_metrics(model, val_images_flat, val_labels)
    print(f"Val   Set -> Accuracy: {val_acc * 100:.2f}%")

    # Evaluate on test set, if exists
    if os.path.exists(test_path):
        test_images, test_labels, _ = torch.load(test_path, weights_only=False)
        test_images_flat = flatten_images(test_images)
        test_acc, _, _ = get_preds_metrics(model, test_images_flat, test_labels)
        print(f"Test  Set -> Accuracy: {test_acc * 100:.2f}%")
    else:
        print("[Info] No test set found, skipping test metrics.")

    print("\nEnd of additional summary.\n")


if __name__ == "__main__":
    main()
