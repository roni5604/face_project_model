"""
basic_nn.py

PURPOSE:
  - Train and compare a basic fully-connected neural network (MLP) with hidden layers, BN, Dropout
    for 7-class facial expression recognition (48x48 grayscale input).
  - This is more advanced than just a single linear layer but lacks convolution.

FUNCTIONS & FLOW:
  - main():
    1) Load data from train_data.pt, val_data.pt
    2) Flatten images to (N, 2304)
    3) Build an MLP: BasicNN with two hidden layers
    4) Train for N epochs, track train/val loss
    5) Print classification report (accuracy, precision, recall, f1) on val set
    6) Optionally evaluate on test set
    7) Plot training curves

HOW TO RUN:
  python basic_nn.py

REQUIREMENTS:
  - Torch, NumPy, sklearn, Matplotlib
  - 'processed_data/train_data.pt', 'processed_data/val_data.pt', 'processed_data/test_data.pt' (optional)
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
# 1) Define the Basic MLP
###############################################################################
class BasicNN(nn.Module):
    """
    A simple MLP with two hidden layers.
    Each hidden layer has:
      - Linear layer
      - BatchNorm1d
      - ReLU
      - Dropout
    Output layer => 7 classes for cross-entropy
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):
        super(BasicNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
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
    images shape: (N,1,48,48)
    flattens => (N, 2304)
    """
    N = images.shape[0]
    return images.view(N, -1)

def train_model(model, loader, criterion, optimizer):
    """
    One epoch of training on 'model' with 'loader'.
    Returns average train loss.
    """
    model.train()
    total_loss = 0
    total_samples = 0
    for data, targets in loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

def validate_model(model, loader, criterion):
    """
    Evaluate model on validation set. Returns:
      - val_loss
      - val_preds
      - val_targets
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    total_samples = 0

    with torch.no_grad():
        for data, targets in loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = total_loss / total_samples
    val_preds = np.concatenate(all_preds)
    val_targets = np.concatenate(all_targets)
    return val_loss, val_preds, val_targets

###############################################################################
# 3) MAIN
###############################################################################
def main():
    print("=========================================================")
    print("  BASIC FULLY CONNECTED NEURAL NETWORK (MLP) FOR EMOTIONS ")
    print("=========================================================\n")

    os.makedirs("results", exist_ok=True)

    train_path = "processed_data/train_data.pt"
    val_path   = "processed_data/val_data.pt"
    test_path  = "processed_data/test_data.pt"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("[Error] Processed train/val data not found.")
        return

    # Load data
    train_images, train_labels, label_to_idx = torch.load(train_path, weights_only=False)
    val_images,   val_labels,   _            = torch.load(val_path, weights_only=False)

    print("[Info] Train set size:", train_labels.size(0))
    print("[Info] Val set size:  ", val_labels.size(0), "\n")

    # Flatten images for MLP
    train_images_flat = flatten_images(train_images)
    val_images_flat   = flatten_images(val_images)

    train_dataset = TensorDataset(train_images_flat, train_labels)
    val_dataset   = TensorDataset(val_images_flat, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_dim   = train_images_flat.shape[1]  # typically 48*48=2304
    hidden_dim1 = 128
    hidden_dim2 = 64
    num_classes = len(label_to_idx)

    model = BasicNN(input_dim, hidden_dim1, hidden_dim2, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_losses = []
    val_losses   = []

    print("[Info] Starting training...\n")
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_preds, val_targets = validate_model(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Classification metrics on validation
    from sklearn.metrics import classification_report, accuracy_score
    report = classification_report(val_targets, val_preds, target_names=label_to_idx.keys(), zero_division=0)
    val_accuracy = accuracy_score(val_targets, val_preds)

    print("\n===============================")
    print(" Validation Classification Report:")
    print("===============================")
    print(report)
    print(f"[Summary] Final Validation Accuracy: {val_accuracy * 100:.2f}%\n")

    # Save results
    results_file = "results/basic_nn_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Final Accuracy: {val_accuracy:.4f}\nClassification Report:\n")
        f.write(report)
    print(f"[Info] Results saved to '{results_file}' for comparison.\n")

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses,   label='Val Loss',   marker='s')
    plt.title('Basic NN (Deeper) Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n[Info] Basic NN training completed.\n")

    # Additional summary
    print("------------------------------------------------------------")
    print("ADDITIONAL SUMMARY: BASIC NN EVALUATION ON TRAIN, VAL, TEST")
    print("------------------------------------------------------------\n")

    def get_preds_metrics(model, data_x, data_y):
        model.eval()
        with torch.no_grad():
            outputs = model(data_x)
            preds = torch.argmax(outputs, dim=1)
        y_true = data_y.numpy()
        y_pred = preds.numpy()

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
        return acc, prec, rec

    train_acc, _, _ = get_preds_metrics(model, train_images_flat, train_labels)
    print(f"Train Set -> Accuracy: {train_acc * 100:.2f}%")

    val_acc, _, _ = get_preds_metrics(model, val_images_flat, val_labels)
    print(f"Val   Set -> Accuracy: {val_acc * 100:.2f}%")

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
