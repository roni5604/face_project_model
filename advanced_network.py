"""
advanced_network.py

PURPOSE:
  - Implements an Advanced Convolutional Neural Network (CNN) for Facial Expression Recognition.
  - Uses Weighted Cross Entropy for class imbalance, Dropout for overfitting reduction,
    StepLR scheduler, and Early Stopping.

HOW TO RUN:
  1) Place train_data.pt and val_data.pt in processed_data/ (optional: test_data.pt as well).
  2) Run: python advanced_network.py
  3) Check "results/advanced_network_results.txt" for final metrics.
  4) A final trained model is saved to "results/advanced_model.pth" for live inference.

NOTES:
  - This script runs on CPU by default (no GPU checks).
  - Data augmentation is not forced here (can be done offline if desired).
  - Weighted Cross Entropy helps address class imbalance. Adjust 'weights' to match your dataset distribution.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

##############################################################################
#                           CNN ARCHITECTURE                                 #
##############################################################################
class AdvancedCNN(nn.Module):
    """
    A deeper CNN with three convolutional layers, each followed by a pooling operation.
    Includes dropout to reduce overfitting, then two fully-connected layers for the final classification.
    """
    def __init__(self, num_classes=7):
        super(AdvancedCNN, self).__init__()
        # 1st block: conv -> ReLU -> pool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd block: conv -> ReLU -> pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd block: conv -> ReLU -> pool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(p=0.4)

        # After 3 pools, 48x48 => 6x6. Flatten dimension = 128 * 6 * 6 = 4608
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Dropout then fully-connected
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

##############################################################################
#                    TRAINING & VALIDATION HELPER FUNCTIONS                  #
##############################################################################
def train_model(model, loader, criterion, optimizer):
    """
    Train the model for one epoch.
    Returns: average training loss
    """
    model.train()
    total_loss = 0.0
    for data, targets in loader:
        outputs = model(data)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)

    return total_loss / len(loader.dataset)

def validate_model(model, loader, criterion):
    """
    Validate the model on a given loader.
    Returns: (val_loss, all_preds, all_targets)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * data.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.numpy())
            all_targets.append(targets.numpy())

    val_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return val_loss, all_preds, all_targets

##############################################################################
#                            MAIN FUNCTION                                   #
##############################################################################
def main():
    print("====================================================================")
    print("   ADVANCED NETWORK (DEEP CNN) FOR FACIAL EXPRESSION RECOGNITION     ")
    print("====================================================================\n")

    # Make output directory
    os.makedirs("results", exist_ok=True)

    # .pt file paths
    train_path = "processed_data/train_data.pt"
    val_path   = "processed_data/val_data.pt"
    test_path  = "processed_data/test_data.pt"  # optional

    # Check if train/val data exist
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("[Error] Train/Val data not found. Stopping.")
        return

    # Weighted Cross Entropy for class imbalance
    # (Adjust these weights as needed for your data)
    weights = torch.tensor([1.0, 5.0, 2.0, 1.0, 1.5, 1.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Load data
    # NOTE: We accept the default 'weights_only=False', trusting local .pt files
    train_images, train_labels, label_to_idx = torch.load(train_path, weights_only=False)
    val_images, val_labels, _                = torch.load(val_path,   weights_only=False)

    test_images, test_labels = None, None
    if os.path.exists(test_path):
        test_images, test_labels, _ = torch.load(test_path, weights_only=False)

    # Dataset info
    print(f"[Info] Train set size: {train_labels.size(0)}")
    print(f"[Info] Val set size:   {val_labels.size(0)}")
    if test_images is not None:
        print(f"[Info] Test set size:  {test_labels.size(0)}")
    print("")

    # Create Datasets and Loaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset   = TensorDataset(val_images,   val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

    test_loader = None
    if test_images is not None:
        test_dataset = TensorDataset(test_images, test_labels)
        test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = AdvancedCNN(num_classes=len(label_to_idx))

    # Optimizer + StepLR
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training config
    num_epochs = 10
    train_losses = []
    val_losses   = []

    # Early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_count = 0
    best_model_state = None

    print("[Info] Starting CNN training...\n")
    for epoch in range(num_epochs):
        # Train
        train_loss = train_model(model, train_loader, criterion, optimizer)

        # Validate
        val_loss, val_preds, val_targets = validate_model(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # LR scheduling step
        scheduler.step()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count= 0
            best_model_state = model.state_dict()
        else:
            patience_count += 1
            if patience_count >= patience:
                print("[Info] Early stopping triggered.")
                break

    # Restore best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final validation metrics
    _, val_preds, val_targets = validate_model(model, val_loader, criterion)
    val_accuracy = accuracy_score(val_targets, val_preds)
    report_val = classification_report(val_targets, val_preds, target_names=label_to_idx.keys(), zero_division=0)

    print("\n===================================================")
    print("  Validation Classification Report (Advanced CNN): ")
    print("===================================================")
    print(report_val)
    print(f"[Summary] Validation Accuracy: {val_accuracy*100:.2f}%")

    # Optional: Evaluate on test set
    if test_loader is not None:
        _, test_preds, test_targets = validate_model(model, test_loader, criterion)
        test_accuracy = accuracy_score(test_targets, test_preds)
        report_test   = classification_report(test_targets, test_preds, target_names=label_to_idx.keys(), zero_division=0)

        print("---------------------------------------------------")
        print(f"[Info] Final Test Accuracy: {test_accuracy*100:.2f}%")
        print("  Test Classification Report:")
        print("---------------------------------------------------")
        print(report_test)
    else:
        print("\n[Warning] No test set found, skipping final test evaluation.")

    # Save results to file
    results_file = "results/advanced_network_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write("Validation Classification Report:\n")
        f.write(report_val)
    print(f"\n[Info] Results saved to '{results_file}'.\n")

    # Save the trained model for live inference
    model_save_path = "results/advanced_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"[Info] Trained model saved to '{model_save_path}'.\n")

    # Plot train vs. val loss
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   marker='s', label='Val Loss')
    plt.title('Advanced CNN Training (Weighted Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("[Info] Advanced CNN training completed.\n")

    # Additional summary: Evaluate entire Train & Val sets
    print("--------------------------------------------------------")
    print("ADDITIONAL SUMMARY: ADVANCED CNN EVALUATION ON ALL SETS")
    print("--------------------------------------------------------\n")

    def evaluate_full_set(model, images, labels):
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
        y_true = labels.numpy()
        y_pred = preds.numpy()

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
        return acc, prec, rec

    train_acc, _, _ = evaluate_full_set(model, train_images, train_labels)
    print(f"Train Set -> Accuracy: {train_acc*100:.2f}%")

    val_acc, _, _ = evaluate_full_set(model, val_images, val_labels)
    print(f"Val   Set -> Accuracy: {val_acc*100:.2f}%")

    if test_images is not None:
        test_acc, _, _ = evaluate_full_set(model, test_images, test_labels)
        print(f"Test  Set -> Accuracy: {test_acc*100:.2f}%")
    else:
        print("[Info] No test set found, skipping test metrics.")

    print("\nEnd of additional summary.\n")


if __name__ == "__main__":
    main()
