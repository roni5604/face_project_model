"""
softmax.py

PURPOSE:
  - Train a single-layer softmax (linear) model for 7-class facial expression classification.
  - Reads train_data.pt, val_data.pt from 'processed_data/' for training/validation.
  - If test_data.pt is present, it also evaluates on the test set; otherwise it skips that step.
  - Logs train & val losses each epoch, plots them, then saves final weights to 'results/softmax_model.pth'.
  - Finally, prints a Validation Classification Report at the end for the val set.

HOW TO RUN:
  python softmax.py

REQUIREMENTS:
  - PyTorch, NumPy, Matplotlib
  - scikit-learn (for classification_report)
  - processed_data/train_data.pt, processed_data/val_data.pt (required)
  - processed_data/test_data.pt (optional, if you want final test evaluation).

GENERAL EXPLANATIONS:
  1) This code is part of a project comparing different models:
     - Baseline (always predict the most common class),
     - Softmax (this script),
     - BasicNN (MLP),
     - Advanced CNN.
  2) A single linear layer (SoftmaxClassifier) effectively performs logistic regression
     for multiple classes. CrossEntropyLoss applies an internal softmax on the outputs.
  3) The goal is to see how this "just a softmax" approach compares to more sophisticated
     neural networks in terms of accuracy and classification metrics.
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# For printing a classification report at the end
from sklearn.metrics import classification_report

##############################################################################
# SUPPRESS FUTUREWARNING ABOUT "weights_only=False" FROM TORCH.LOAD
# Explanation: we load .pt files that contain entire tuples, not just weights,
# so we ignore the future warning about 'weights_only' in PyTorch.
##############################################################################
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

##############################################################################
# 1) SOFTMAX MODEL DEFINITION
##############################################################################
class SoftmaxClassifier(nn.Module):
    """
    A simple linear model for multi-class classification:
      - Input dimension: 48*48 = 2304 (flattened grayscale image).
      - Output dimension: 7 (classes).
    The final CrossEntropyLoss call internally applies softmax to these outputs,
    so we only need to return raw logits from the linear layer.

    Purpose in this project:
    - Compare how a purely linear approach (just one linear layer) performs
      vs. more advanced models like BasicNN or a CNN.
    """
    def __init__(self, input_dim=48*48, num_classes=7):
        super().__init__()
        # single linear layer from input_dim => num_classes
        # e.g., 2304 => 7
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        x shape: (N,1,48,48).
        Flatten => shape (N,2304) => linear => shape (N,7).
        The outputs are 'logits' for each of the 7 classes.
        """
        batch_size = x.size(0)
        # Flatten the 48x48 image into a single vector of length 2304
        x = x.view(batch_size, -1)
        logits = self.linear(x)
        return logits

##############################################################################
# 2) TRAIN & VALIDATION FUNCTIONS
##############################################################################
def train_one_epoch(model, loader, criterion, optimizer):
    """
    Train the model for ONE epoch using 'loader'.
    Returns average train loss for that epoch.

    Steps:
      1) Put model in train mode => model.train()
      2) For each batch, do forward pass, compute cross-entropy loss,
         backprop, and optimize.
      3) Accumulate total loss to return an average across the entire dataset.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)            # forward pass
        loss = criterion(outputs, labels)  # cross-entropy
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss

@torch.no_grad()
def validate_one_epoch(model, loader, criterion):
    """
    Evaluate model on 'loader'.
    Returns average validation loss across the entire dataset.

    Steps:
      1) model.eval() to disable dropout, adjust BatchNorm in eval mode
      2) forward pass, compute loss for each batch
      3) accumulate total to produce an average
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss

##############################################################################
# 3) CLASSIFICATION REPORT ON VAL SET
##############################################################################
@torch.no_grad()
def print_val_classification_report(model, loader, num_classes=7):
    """
    Generate a classification report for the entire validation set.
    We'll gather all predictions, gather all ground-truth,
    then use sklearn.metrics.classification_report to display
    precision/recall/f1 for each class + overall stats.

    :param model: trained Softmax model
    :param loader: validation DataLoader
    :param num_classes: number of classes (default=7)
    :return: prints the classification report, returns final accuracy as float

    Explanation:
      - We do a forward pass for each batch.
      - Argmax on the logits gives the predicted class index.
      - Compare to the real label, gather them all, then call classification_report.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

    # Combine predictions and labels from multiple batches
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Create a classification report with digits=3 (3 decimal places)
    report = classification_report(
        all_labels.cpu().numpy(),
        all_preds.cpu().numpy(),
        labels=list(range(num_classes)),  # e.g., [0,1,2,3,4,5,6]
        digits=3
    )

    # Compute overall accuracy manually
    correct_count = (all_preds == all_labels).sum().item()
    total_count   = all_labels.size(0)
    final_accuracy = correct_count / total_count if total_count > 0 else 0.0

    print("\n===============================")
    print(" Validation Classification Report:")
    print("===============================")
    print(report)
    print(f"[Summary] Final Validation Accuracy: {final_accuracy*100:.2f}%\n")

    return final_accuracy

##############################################################################
# 4) MAIN TRAINING/TEST + PLOTTING
##############################################################################
def main():
    """
    Steps:
      1) Confirm train_data.pt, val_data.pt exist.
      2) Load them, create DataLoaders.
      3) Build SoftmaxClassifier, define CrossEntropy, optimizer=SGD.
      4) For each epoch: train_one_epoch + validate_one_epoch => track losses.
      5) Plot final losses, save model.
      6) Evaluate on test_data.pt if available.
      7) Print a classification report on the validation set.

    The code compares how well a single linear layer (Softmax) does
    relative to baseline, MLP, or advanced CNN in the overall project.
    """

    # File paths for .pt data
    train_path = "processed_data/train_data.pt"
    val_path   = "processed_data/val_data.pt"
    test_path  = "processed_data/test_data.pt"

    # Check if train/val .pt files exist
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("[Error] Missing train_data.pt or val_data.pt in 'processed_data/'. Exiting.")
        return

    # 1) Load data
    print("[Info] Loading train/val data from .pt files...")
    # Each .pt file typically has (images_tensor, labels_tensor, label_to_idx)
    train_images, train_labels, label_to_idx = torch.load(train_path)
    val_images,   val_labels,   _            = torch.load(val_path)

    # Build TensorDatasets for train & val
    train_ds = TensorDataset(train_images, train_labels)
    val_ds   = TensorDataset(val_images,   val_labels)

    # Create DataLoaders to iterate in batches
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2)

    print(f"[Info] train_ds={len(train_ds)}, val_ds={len(val_ds)}")
    print(f"[Info] Train batch_size={batch_size}, Val batch_size={batch_size*2}")

    # 2) Create Softmax model: Flatten(48x48=2304) => 7 classes
    input_dim = 48*48
    num_classes = len(label_to_idx)  # e.g., 7
    model_softmax = SoftmaxClassifier(input_dim=input_dim, num_classes=num_classes)
    print("[Info] Created SoftmaxClassifier with single linear layer.")

    # 3) Define loss & optimizer
    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_softmax.parameters(), lr=lr)
    print(f"[Info] Using CrossEntropyLoss + SGD(lr={lr})")

    # 4) Training loop across multiple epochs
    epochs = 10
    train_losses = []
    val_losses   = []

    print("[Info] Starting training...\n")
    for epoch in range(epochs):
        # 4a) Train for one epoch
        train_loss = train_one_epoch(model_softmax, train_loader, criterion, optimizer)
        # 4b) Validate on val set
        val_loss   = validate_one_epoch(model_softmax, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] => "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")

    print("\n[Info] Training completed. Plotting losses...")

    # 5) Plot train vs val losses
    plt.figure()
    plt.plot(range(1,epochs+1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1,epochs+1), val_losses,   marker='s', label='Val Loss')
    plt.title("Softmax Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save final model
    os.makedirs("results", exist_ok=True)
    model_path = "results/softmax_model.pth"
    torch.save(model_softmax.state_dict(), model_path)
    print(f"[Info] Model weights saved to '{model_path}'")

    # 6) Evaluate on test set if test_data.pt is found
    if os.path.exists(test_path):
        print("\n[Info] Found test_data.pt. Evaluating on test set now.")
        test_images, test_labels, _ = torch.load(test_path)
        test_ds = TensorDataset(test_images, test_labels)
        test_loader = DataLoader(test_ds, batch_size=batch_size*2)

        # test_loss: use validate_one_epoch for measuring final average loss
        test_loss = validate_one_epoch(model_softmax, test_loader, criterion)
        print(f"[Test] Loss: {test_loss:.4f}")

        # compute test accuracy
        correct = 0
        total   = 0
        model_softmax.eval()
        with torch.no_grad():
            for imgs, labs in test_loader:
                outputs = model_softmax(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labs).sum().item()
                total   += labs.size(0)
        test_acc = correct / total if total else 0
        print(f"[Test] Accuracy: {test_acc:.4f}")
    else:
        print("\n[Warning] test_data.pt not found, skipping test evaluation.")

    # 7) Print a classification report on the validation set
    #    This will show precision/recall/f1 for each class, plus overall accuracy
    print_val_classification_report(model_softmax, val_loader, num_classes=num_classes)

    print("[Info] Done. The Softmax classifier is fully trained if test set was available.")


if __name__ == "__main__":
    main()
