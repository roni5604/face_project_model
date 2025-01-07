"""
======================================================================
    SOFTMAX (MULTICLASS LOGISTIC REGRESSION) FOR FACIAL EXPRESSION
======================================================================

ADDED:
  - A second summary block that evaluates train, val, (and test if exists)
    altogether, printing accuracy, precision, recall for each set.

NOTE:
  - We do not alter existing code; we only add new lines at the end.
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader

class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def flatten_images(images):
    N = images.shape[0]
    return images.view(N, -1)

def train_model(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for data, targets in loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)

def validate_model(model, loader, criterion):
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
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return val_loss, all_preds, all_targets

def main():
    print("======================================================================")
    print("    SOFTMAX (MULTICLASS LOGISTIC REGRESSION) FOR FACIAL EXPRESSION    ")
    print("======================================================================\n")

    os.makedirs("results", exist_ok=True)
    train_path = "processed_data/train_data.pt"
    val_path   = "processed_data/val_data.pt"
    test_path  = "processed_data/test_data.pt"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("[Error] Processed train/val data not found. Run dataset_preparation.py first.")
        return

    train_images, train_labels, label_to_idx = torch.load(train_path, weights_only=False)
    val_images,   val_labels,   _            = torch.load(val_path,   weights_only=False)

    print("[Info] Train set size:", train_labels.size(0))
    print("[Info] Val set size:  ", val_labels.size(0), "\n")

    train_images_flat = flatten_images(train_images)
    val_images_flat   = flatten_images(val_images)

    train_dataset = TensorDataset(train_images_flat, train_labels)
    val_dataset   = TensorDataset(val_images_flat,   val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

    input_dim   = train_images_flat.shape[1]  # 2304
    num_classes = len(label_to_idx)
    learning_rate = 0.01
    num_epochs    = 10

    model     = SoftmaxRegression(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses   = []

    print("[Info] Starting training...\n")

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_preds, val_targets = validate_model(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    val_accuracy = accuracy_score(val_targets, val_preds)
    report = classification_report(val_targets, val_preds, target_names=label_to_idx.keys())

    print("\n===============================")
    print(" Validation Classification Report:")
    print("===============================")
    print(report)
    print(f"[Summary] Final Validation Accuracy: {val_accuracy*100:.2f}%\n")

    # Save results
    results_file = "results/softmax_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Final Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Classification Report:\n{report}\n")
    print(f"[Info] Results saved to '{results_file}' for later comparison.\n")

    print("Plotting the training/validation loss curves...\n")
    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs+1), val_losses,   label='Val Loss',   marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Softmax Regression Training')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("[Info] Softmax training completed.\n")

    # -------------------------------------------------------------------
    # ADDITIONAL SUMMARY BLOCK: Evaluate on train, val, test for P/R/A
    # -------------------------------------------------------------------
    print("------------------------------------------------------------")
    print("ADDITIONAL SUMMARY: SOFTMAX EVALUATION ON TRAIN, VAL, TEST")
    print("------------------------------------------------------------\n")

    # We'll define a helper to get predictions & metrics:
    def get_predictions_and_metrics(model, images, labels):
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
        y_true = labels.numpy()
        y_pred = preds.numpy()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec = recall_score(y_true, y_pred, average=None, zero_division=0)
        return acc, prec, rec

    # Evaluate on train set
    train_acc, train_prec, train_rec = get_predictions_and_metrics(model, train_images_flat, train_labels)
    print(f"Train Set -> Accuracy: {train_acc*100:.2f}%")

    # Evaluate on val set
    val_acc, val_prec, val_rec = get_predictions_and_metrics(model, val_images_flat, val_labels)
    print(f"Val   Set -> Accuracy: {val_acc*100:.2f}%")

    # Evaluate on test set if it exists
    if os.path.exists(test_path):
        # Load test
        test_images, test_labels, _ = torch.load(test_path, weights_only=False)
        test_images_flat = flatten_images(test_images)
        test_acc, _, _ = get_predictions_and_metrics(model, test_images_flat, test_labels)
        print(f"Test  Set -> Accuracy: {test_acc*100:.2f}%")
    else:
        print("[Info] No test set found, skipping test metrics.")

    print("\nPrecision and Recall arrays are also computed. You can print them out if needed.")
    print("End of additional summary.\n")

if __name__ == "__main__":
    main()
