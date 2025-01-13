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
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# For printing a classification report
from sklearn.metrics import classification_report

##############################################################################
# SUPPRESS FUTUREWARNING ABOUT "weights_only=False" FROM TORCH.LOAD
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
    CrossEntropyLoss automatically applies softmax internally.
    """
    def __init__(self, input_dim=48*48, num_classes=7):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        x shape: (N,1,48,48).
        Flatten => shape (N,2304) => linear => shape (N,7).
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)   # flatten (1,48,48)->(1,2304)
        logits = self.linear(x)
        return logits

##############################################################################
# 2) TRAIN & VALIDATION FUNCTIONS
##############################################################################
def train_one_epoch(model, loader, criterion, optimizer):
    """
    Train the model for ONE epoch using 'loader'.
    Returns average train loss for that epoch.
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
    Returns average validation loss.
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
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

    # Combine all predictions, labels
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute classification report
    # By default, classification_report shows numeric labels.
    # You can define label names if you have them.
    report = classification_report(
        all_labels.cpu().numpy(),
        all_preds.cpu().numpy(),
        labels=list(range(num_classes)),  # e.g., 0..6
        digits=3
    )

    # Compute overall accuracy
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
      6) (Optional) Evaluate on test_data.pt if available.
      7) Print a classification report on the validation set.
    """
    # File paths
    train_path = "processed_data/train_data.pt"
    val_path   = "processed_data/val_data.pt"
    test_path  = "processed_data/test_data.pt"

    # Check existence
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("[Error] Missing train_data.pt or val_data.pt in 'processed_data/'. Exiting.")
        return

    # 1) Load data
    print("[Info] Loading train/val data from .pt files...")
    train_images, train_labels, label_to_idx = torch.load(train_path)
    val_images,   val_labels,   _            = torch.load(val_path)

    # Build datasets, loaders
    train_ds = TensorDataset(train_images, train_labels)
    val_ds   = TensorDataset(val_images,   val_labels)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2)

    print(f"[Info] train_ds={len(train_ds)}, val_ds={len(val_ds)}")
    print(f"[Info] Train batch_size={batch_size}, Val batch_size={batch_size*2}")

    # 2) Create Softmax model
    input_dim = 48*48  # flatten grayscale 48x48
    num_classes = len(label_to_idx)  # typically 7
    model_softmax = SoftmaxClassifier(input_dim=input_dim, num_classes=num_classes)
    print("[Info] Created SoftmaxClassifier with single linear layer.")

    # 3) Define loss & optimizer
    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_softmax.parameters(), lr=lr)
    print(f"[Info] Using CrossEntropyLoss + SGD(lr={lr})")

    # 4) Training loop
    epochs = 10
    train_losses = []
    val_losses   = []

    print("[Info] Starting training...\n")
    for epoch in range(epochs):
        # train
        train_loss = train_one_epoch(model_softmax, train_loader, criterion, optimizer)
        # validate
        val_loss   = validate_one_epoch(model_softmax, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] => "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")

    print("\n[Info] Training completed. Plotting losses...")

    # 5) Plot train vs val losses
    import matplotlib.pyplot as plt
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

    # 6) Optional test set
    if os.path.exists(test_path):
        print("\n[Info] Found test_data.pt. Evaluating on test set now.")
        test_images, test_labels, _ = torch.load(test_path)
        test_ds = TensorDataset(test_images, test_labels)
        test_loader = DataLoader(test_ds, batch_size=batch_size*2)

        # test loss
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

    # 7) Print classification report on the validation set
    #    (We do this after training is complete.)
    print_val_classification_report(model_softmax, val_loader, num_classes=num_classes)

    print("[Info] Done. The Softmax classifier is fully trained if test set was available.")


if __name__ == "__main__":
    main()
