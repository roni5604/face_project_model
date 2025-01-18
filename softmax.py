"""
softmax.py

PURPOSE:
  - Train a single-layer softmax (linear) model for 7-class facial expression classification.
  - Reads train_data.pt, val_data.pt from 'processed_data/' for training/validation.
  - If test_data.pt is present, evaluates it too. Logs train & val losses each epoch,
    plots them, then saves final weights.
  - Prints a Validation Classification Report with Accuracy, Precision, Recall, F1.

USAGE:
  python softmax.py

REQUIREMENTS:
  - PyTorch, NumPy, Matplotlib, sklearn
  - processed_data/train_data.pt, processed_data/val_data.pt, optionally test_data.pt
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Suppress warnings for torch.load 'weights_only=False'
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

class SoftmaxClassifier(nn.Module):
    """
    A simple linear model for multi-class classification:
      - Flatten 48x48 => 2304
      - Single linear => 7 outputs.
    CrossEntropyLoss will do the softmax behind the scenes.
    """
    def __init__(self, input_dim=48*48, num_classes=7):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # flatten
        logits = self.linear(x)
        return logits

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

@torch.no_grad()
def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

@torch.no_grad()
def print_val_classification_report(model, loader, num_classes=7):
    model.eval()
    all_preds = []
    all_labels = []
    for images, labels in loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # classification_report
    report = classification_report(
        all_labels.cpu().numpy(),
        all_preds.cpu().numpy(),
        labels=list(range(num_classes)),
        digits=3
    )

    correct_count = (all_preds == all_labels).sum().item()
    total_count   = all_labels.size(0)
    final_accuracy = correct_count / total_count if total_count else 0

    print("\n===============================")
    print(" Validation Classification Report:")
    print("===============================")
    print(report)
    print(f"[Summary] Final Validation Accuracy: {final_accuracy*100:.2f}%\n")

def main():
    train_path = "processed_data/train_data.pt"
    val_path   = "processed_data/val_data.pt"
    test_path  = "processed_data/test_data.pt"

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("[Error] Missing train_data.pt or val_data.pt in 'processed_data/'. Exiting.")
        return

    print("[Info] Loading train/val data from .pt files...")
    train_images, train_labels, label_to_idx = torch.load(train_path)
    val_images,   val_labels,   _            = torch.load(val_path)

    train_ds = TensorDataset(train_images, train_labels)
    val_ds   = TensorDataset(val_images,   val_labels)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size*2)

    print(f"[Info] train_ds={len(train_ds)}, val_ds={len(val_ds)}")
    print(f"[Info] Train batch_size={batch_size}, Val batch_size={batch_size*2}")

    input_dim = 48*48
    num_classes = len(label_to_idx)
    model_softmax = SoftmaxClassifier(input_dim=input_dim, num_classes=num_classes)
    print("[Info] Created SoftmaxClassifier with single linear layer.")

    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_softmax.parameters(), lr=lr)
    print(f"[Info] Using CrossEntropyLoss + SGD(lr={lr})")

    epochs = 10
    train_losses = []
    val_losses   = []

    print("[Info] Starting training...\n")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model_softmax, train_loader, criterion, optimizer)
        val_loss   = validate_one_epoch(model_softmax, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] => "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")

    print("\n[Info] Training completed. Plotting losses...")

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

    os.makedirs("results", exist_ok=True)
    model_path = "results/softmax_model.pth"
    torch.save(model_softmax.state_dict(), model_path)
    print(f"[Info] Model weights saved to '{model_path}'")

    # Evaluate on test set if available
    if os.path.exists(test_path):
        print("\n[Info] Found test_data.pt. Evaluating on test set now.")
        test_images, test_labels, _ = torch.load(test_path)
        test_ds = TensorDataset(test_images, test_labels)
        test_loader = DataLoader(test_ds, batch_size=batch_size*2)

        test_loss = validate_one_epoch(model_softmax, test_loader, criterion)
        print(f"[Test] Loss: {test_loss:.4f}")

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

    # Print classification report on the validation set
    print_val_classification_report(model_softmax, val_loader, num_classes=num_classes)

    print("[Info] Done. The Softmax classifier is fully trained if test set was available.")


if __name__ == "__main__":
    main()
