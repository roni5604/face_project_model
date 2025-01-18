"""
advanced_network.py

PURPOSE:
  - Demonstrates an advanced Convolutional Neural Network (CNN) architecture for
    7-class facial expression recognition using PyTorch.
  - Architecture: multiple Conv + BatchNorm + ReLU + MaxPool + Dropout blocks,
    then flattening followed by two fully connected layers (each with BN, Dropout),
    and a final 7-class output layer.
  - Outputs standard training logs (train_loss, val_loss, val_acc per epoch) and
    a final classification report (precision, recall, F1) on the validation set.
  - Saves final model weights to 'results/advanced_model.pth' so that 'live_inference.py'
    can perform real-time classification if desired.

HOW TO RUN:
  python advanced_network.py

REQUIREMENTS:
  - PyTorch, NumPy, Matplotlib, scikit-learn
  - The processed .pt files in 'processed_data/', namely:
       * train_data.pt  (train_imgs, train_labels, label_to_idx)
       * val_data.pt    (val_imgs, val_labels, same label_to_idx)
       * (Optional) test_data.pt
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

##############################################################################
# 0) SUPPRESS FUTUREWARNING ABOUT 'weights_only=False'
##############################################################################
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

##############################################################################
# 1) HELPER FUNCTION TO COMPUTE BATCH ACCURACY
##############################################################################
def compute_accuracy(batch_outputs, batch_labels):
    """
    Compute classification accuracy for a single batch.

    :param batch_outputs: (batch_size, num_classes) raw logits
    :param batch_labels : (batch_size,) integer class labels
    :return: torch scalar in [0..1] => fraction of correct predictions
    """
    _, predicted = torch.max(batch_outputs, dim=1)
    correct = (predicted == batch_labels).sum().item()
    return torch.tensor(correct / len(batch_labels))

##############################################################################
# 2) BASE CLASS FOR TRAINING/VALIDATION STEPS
##############################################################################
class CNNTrainingBase(nn.Module):
    """
    Provides standard methods for training and validation in each epoch:
      - training_step(batch) => CrossEntropy
      - validation_step(batch) => dict of {'val_loss', 'val_acc'}
      - validation_epoch_end(batch_outputs) => aggregates over batches
      - epoch_end(epoch, result) => prints epoch summary
    """

    def training_step(self, batch):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        acc  = compute_accuracy(logits, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, batch_outputs):
        losses = [x['val_loss'] for x in batch_outputs]
        avg_loss = torch.stack(losses).mean()
        accs = [x['val_acc'] for x in batch_outputs]
        avg_acc = torch.stack(accs).mean()
        return {'val_loss': avg_loss.item(), 'val_acc': avg_acc.item()}

    def epoch_end(self, epoch_idx, epoch_result):
        print(f"Epoch [{epoch_idx+1}], "
              f"train_loss: {epoch_result['train_loss']:.4f}, "
              f"val_loss: {epoch_result['val_loss']:.4f}, "
              f"val_acc: {epoch_result['val_acc']:.4f}")

##############################################################################
# 3) ADVANCED CNN MODEL (CONV + BN + POOL + DROPOUT + FC)
##############################################################################
class AdvancedEmotionCNN(CNNTrainingBase):
    """
    A more advanced CNN architecture for 7-class emotion classification:
      - 4 convolution blocks
      - Flatten
      - 2 fully connected layers
      - Final 7-class output
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.25),

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.25),

            # Conv Block 3
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.25),

            # Conv Block 4
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.25),

            # Flatten
            nn.Flatten(),
            nn.Linear(512*3*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(512, 7)  # 7 emotion classes
        )

    def forward(self, x):
        return self.network(x)

##############################################################################
# 4) TRAIN/EVALUATE LOOPS
##############################################################################
@torch.no_grad()
def evaluate_model(model, loader):
    """
    Evaluate the model on 'loader': aggregates val_loss & val_acc for entire dataset.
    Returns dict: {'val_loss', 'val_acc'}.
    """
    model.eval()
    outputs = []
    for batch in loader:
        step_dict = model.validation_step(batch)
        outputs.append(step_dict)
    return model.validation_epoch_end(outputs)

def train_model(num_epochs, lr, model, train_loader, val_loader, optimizer_func=torch.optim.SGD):
    """
    Trains the model for 'num_epochs' with the given train_loader/val_loader.
    :param num_epochs   : number of epochs
    :param lr           : learning rate
    :param model        : instance of AdvancedEmotionCNN
    :param train_loader : DataLoader for training
    :param val_loader   : DataLoader for validation
    :param optimizer_func: which optimizer to use (default SGD)
    Returns 'history': a list of dicts (one per epoch) containing
         {train_loss, val_loss, val_acc}.
    """
    history = []
    optimizer = optimizer_func(model.parameters(), lr=lr)

    for epoch_i in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        val_metrics = evaluate_model(model, val_loader)
        avg_train_loss = torch.stack(train_losses).mean().item()
        val_metrics['train_loss'] = avg_train_loss

        # Print epoch summary
        model.epoch_end(epoch_i, val_metrics)
        history.append(val_metrics)

    return history

def plot_validation_accuracy(history):
    """
    Plots val_acc vs. epochs from 'history'.
    """
    accuracies = [h['val_acc'] for h in history]
    plt.figure()
    plt.plot(accuracies, '-x')
    plt.title("Validation Accuracy vs. Epochs (Advanced CNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

def plot_train_vs_val_loss(history):
    """
    Plots training loss vs validation loss from 'history'.
    """
    train_losses = [h['train_loss'] for h in history]
    val_losses   = [h['val_loss']   for h in history]
    plt.figure()
    plt.plot(train_losses, '-bx', label='Train Loss')
    plt.plot(val_losses,   '-rx', label='Val Loss')
    plt.title("Training vs Validation Loss (Advanced CNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

##############################################################################
# 5) CLASSIFICATION REPORT ON VALIDATION
##############################################################################
@torch.no_grad()
def print_val_classification_report(model, loader, label_to_idx):
    """
    Print precision, recall, and f1 for each class on the entire validation set.

    Steps:
      1) model.eval()
      2) forward -> argmax for each batch
      3) gather predictions, compare to labels
      4) classification_report from sklearn
    """
    model.eval()
    all_preds = []
    all_labels = []
    for images, labels in loader:
        logits = model(images)
        preds  = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # We can pass 'labels=list(range(len(label_to_idx)))' to ensure indices go 0..6
    report = classification_report(y_true, y_pred,
                                   labels=list(range(len(label_to_idx))),
                                   digits=3)
    # overall accuracy
    correct = (y_pred == y_true).sum()
    total   = y_true.shape[0]
    final_acc = correct / total if total else 0

    print("\n======================================")
    print(" Validation Classification Report:")
    print("======================================")
    print(report)
    print(f"[Summary] Validation Accuracy (from this report): {final_acc*100:.2f}%\n")

##############################################################################
# 6) EVALUATE ENTIRE DATASET ACCURACY
##############################################################################
@torch.no_grad()
def compute_full_dataset_accuracy(model, images, labels):
    """
    Evaluate model's accuracy on the entire dataset in one shot (no mini-batches).
    images: (N,1,48,48)
    labels: (N,)
    Returns float in [0,1].
    """
    model.eval()
    logits = model(images)
    _, preds = torch.max(logits, dim=1)
    correct = (preds == labels).sum().item()
    total   = labels.size(0)
    return correct / total

##############################################################################
# 7) MAIN
##############################################################################
def main():
    """
    1) Load train_data.pt, val_data.pt from 'processed_data/'.
    2) Create DataLoaders + AdvancedEmotionCNN.
    3) Train for a certain number of epochs, printing epoch-based logs.
    4) Plot final val_acc and train/val loss curves.
    5) Print classification report on the validation set (precision, recall, f1).
    6) Evaluate entire train and val sets for overall accuracy.
    7) Save final model to 'results/advanced_model.pth'.
    """
    train_pt = "processed_data/train_data.pt"
    val_pt   = "processed_data/val_data.pt"
    test_pt  = "processed_data/test_data.pt"

    if not (os.path.exists(train_pt) and os.path.exists(val_pt)):
        print("[Error] Missing train_data.pt or val_data.pt in 'processed_data/'.")
        return

    # Load data
    print("[Info] Loading training & validation data from .pt files...")
    train_imgs, train_lbls, label_to_idx = torch.load(train_pt)
    val_imgs,   val_lbls,   _            = torch.load(val_pt)

    # Build DataLoaders
    train_ds = TensorDataset(train_imgs, train_lbls)
    val_ds   = TensorDataset(val_imgs,  val_lbls)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=128)

    # Create advanced CNN
    model = AdvancedEmotionCNN()
    print("[Info] Created advanced CNN with multiple convolution blocks.\n")

    # Train
    epochs = 10
    lr     = 0.001
    print(f"[Info] Training for {epochs} epochs with lr={lr} (Adam)...\n")
    history = train_model(epochs, lr, model, train_dl, val_dl, optimizer_func=torch.optim.Adam)

    # Plot accuracy and losses
    plot_validation_accuracy(history)
    plot_train_vs_val_loss(history)

    # Print classification report on val set
    print_val_classification_report(model, val_dl, label_to_idx)

    # Evaluate entire train set
    train_acc = compute_full_dataset_accuracy(model, train_imgs, train_lbls)
    # Evaluate entire val set
    val_acc   = compute_full_dataset_accuracy(model, val_imgs, val_lbls)

    print("[Info] Full-set accuracies (no mini-batches):")
    print(f"Train Set -> Accuracy: {train_acc*100:.2f}%")
    print(f"Val   Set -> Accuracy: {val_acc*100:.2f}%")


    # Save final model
    os.makedirs("results", exist_ok=True)
    final_model_path = "results/advanced_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"[Info] Advanced CNN model saved to '{final_model_path}'")


if __name__ == "__main__":
    main()
