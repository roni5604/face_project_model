"""
advanced_network.py

PURPOSE:
  - Demonstrates an advanced Convolutional Neural Network (CNN) architecture
    for 7-class facial expression recognition using PyTorch.
  - The architecture includes multiple Convolution + BatchNorm + ReLU + MaxPool + Dropout blocks,
    followed by flattening and two fully-connected layers with additional dropout.
  - Outputs standard training logs (epoch-based) and final accuracy on the
    entire training and validation sets.
  - Saves final weights to 'results/advanced_model.pth' so that a similarly
    defined model in live_inference.py can be used for real-time classification.

REQUIREMENTS:
  - PyTorch, NumPy, Matplotlib
  - 'processed_data/train_data.pt' and 'processed_data/val_data.pt' containing:
      (images_tensor, labels_tensor, label_to_idx).
    Each image: shape (N,1,48,48), each label: shape (N,).
    7 classes in total.
  - Optional: 'processed_data/test_data.pt' if you want further testing (not done here).

HOW TO RUN:
  python advanced_network.py
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

##############################################################################
# 0) SUPPRESS WARNINGS ABOUT torch.load (weights_only=False)
##############################################################################
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

##############################################################################
# 1) HELPER FUNCTIONS AND BASE CLASSES
##############################################################################

def compute_accuracy(batch_outputs, batch_labels):
    """
    Compute classification accuracy for a single batch.
    :param batch_outputs: shape (batch_size, num_classes)
    :param batch_labels: shape (batch_size)
    :return: torch scalar in [0..1] representing fraction of correct predictions
    """
    _, predicted_indices = torch.max(batch_outputs, dim=1)
    correct_count = (predicted_indices == batch_labels).sum().item()
    accuracy_fraction = correct_count / len(batch_labels)
    return torch.tensor(accuracy_fraction)


class CNNTrainingBase(nn.Module):
    """
    A base class that defines standard methods for training and validation steps.
    This helps keep code organized.
    """

    def training_step(self, batch):
        """
        One forward pass + cross-entropy for a training batch.
        The result is the training loss (a scalar).
        """
        images, labels = batch
        model_outputs = self(images)
        loss = F.cross_entropy(model_outputs, labels)
        return loss

    def validation_step(self, batch):
        """
        Returns a dict for each validation batch:
           {'val_loss': <loss>, 'val_acc': <accuracy>}
        We'll combine these in validation_epoch_end.
        """
        images, labels = batch
        model_outputs = self(images)
        loss = F.cross_entropy(model_outputs, labels)
        acc  = compute_accuracy(model_outputs, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, batch_outputs):
        """
        Combine the results from all val batches:
        we average val_loss and val_acc across the entire val set.
        """
        losses = [x['val_loss'] for x in batch_outputs]
        val_loss_avg = torch.stack(losses).mean()
        accs   = [x['val_acc'] for x in batch_outputs]
        val_acc_avg  = torch.stack(accs).mean()
        return {'val_loss': val_loss_avg.item(), 'val_acc': val_acc_avg.item()}

    def epoch_end(self, epoch_idx, epoch_result):
        """
        Print a summary line: train_loss, val_loss, val_acc
        """
        print(f"Epoch [{epoch_idx+1}], "
              f"train_loss: {epoch_result['train_loss']:.4f}, "
              f"val_loss: {epoch_result['val_loss']:.4f}, "
              f"val_acc: {epoch_result['val_acc']:.4f}")

##############################################################################
# 2) ADVANCED CNN MODEL
##############################################################################
class AdvancedEmotionCNN(CNNTrainingBase):
    """
    An advanced CNN architecture for 7-class classification (facial expressions).
    The structure:

    -- 1st block:
         Conv2d(1->64, kernel_size=5, padding=2),
         BN(64), ReLU, MaxPool(2,2), Dropout(0.25)
    -- 2nd block:
         Conv2d(64->128, kernel_size=3, padding=1),
         BN(128), ReLU, MaxPool(2,2), Dropout(0.25)
    -- 3rd block:
         Conv2d(128->512, kernel_size=3, padding=1),
         BN(512), ReLU, MaxPool(2,2), Dropout(0.25)
    -- 4th block:
         Conv2d(512->512, kernel_size=3, padding=1),
         BN(512), ReLU, MaxPool(2,2), Dropout(0.25)
    -- Flatten => shape (512*3*3 = 4608)
    -- FC1: (4608->256), BN(256), ReLU, Dropout(0.25)
    -- FC2: (256->512), BN(512), ReLU, Dropout(0.25)
    -- final: (512->7)
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # block 1 => kernel=5 for the first conv
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # block 2 => kernel=3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # block 3 => kernel=3
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # block 4 => kernel=3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # flatten
            nn.Flatten(),

            # FC1 => (4608->256)
            nn.Linear(512*3*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            # FC2 => (256->512)
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            # Output => 7 classes
            nn.Linear(512, 7)
        )

    def forward(self, x):
        return self.network(x)


##############################################################################
# 3) TRAIN/EVAL LOOPS
##############################################################################
@torch.no_grad()
def evaluate_model(model, data_loader):
    """
    Evaluate the model on the entire data_loader, computing average val_loss & val_acc.
    Returns a dict: {'val_loss', 'val_acc'}.
    """
    model.eval()
    all_outputs = []
    for batch in data_loader:
        step_out = model.validation_step(batch)
        all_outputs.append(step_out)
    return model.validation_epoch_end(all_outputs)

def train_model(num_epochs, learning_rate, model, train_loader, val_loader, optimizer_func=torch.optim.SGD):
    """
    Train the model for 'num_epochs', logging each epoch's train_loss, val_loss, val_acc.
    Returns 'history': a list of dicts with {'train_loss', 'val_loss', 'val_acc'}.
    """
    history = []
    optimizer = optimizer_func(model.parameters(), lr=learning_rate)

    for epoch_i in range(num_epochs):
        # training phase
        model.train()
        train_losses_this_epoch = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses_this_epoch.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # validation phase
        val_results = evaluate_model(model, val_loader)
        avg_train_loss = torch.stack(train_losses_this_epoch).mean().item()
        val_results['train_loss'] = avg_train_loss

        # log
        model.epoch_end(epoch_i, val_results)
        history.append(val_results)

    return history


##############################################################################
# 4) PLOTTING
##############################################################################
def plot_validation_accuracy(history):
    """
    Plot the val_acc across epochs from 'history'.
    'history' is a list of dicts, each containing 'val_acc', 'val_loss', 'train_loss'.
    """
    accuracies = [h['val_acc'] for h in history]
    plt.figure()
    plt.plot(accuracies, '-x')
    plt.title("Validation Accuracy vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

def plot_train_vs_val_loss(history):
    """
    Plot training vs validation loss across epochs from 'history'.
    """
    train_losses = [h['train_loss'] for h in history]
    val_losses   = [h['val_loss']   for h in history]

    plt.figure()
    plt.plot(train_losses, '-bx', label='Train Loss')
    plt.plot(val_losses,   '-rx', label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


##############################################################################
# 5) ACCURACY ON FULL SET
##############################################################################
@torch.no_grad()
def compute_full_dataset_accuracy(model, images, labels):
    """
    Evaluate model on the entire dataset in one pass.
    images: shape (N,1,48,48)
    labels: shape (N,).
    returns a float in [0..1].
    """
    model.eval()
    outputs = model(images)  # shape (N,7)
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total   = labels.size(0)
    return correct / total

##############################################################################
# 6) MAIN
##############################################################################
def main():
    """
    Steps:
      1) Load train_data.pt, val_data.pt from 'processed_data/'
      2) Create DataLoaders
      3) Build advanced CNN with first conv kernel=5, second conv kernel=3, etc.
      4) Train for 10 epochs, record epoch logs
      5) Plot final val_acc and train/val loss
      6) Save model => 'results/advanced_model.pth'
      7) Print final train, val set accuracy
    """
    train_pt = "processed_data/train_data.pt"
    val_pt   = "processed_data/val_data.pt"
    test_pt  = "processed_data/test_data.pt"

    if not (os.path.exists(train_pt) and os.path.exists(val_pt)):
        print("[Error] Missing train_data.pt or val_data.pt!")
        return

    # load
    print("[Info] Loading training & validation data from .pt ...")
    train_imgs, train_lbls, label_to_idx = torch.load(train_pt)
    val_imgs,   val_lbls,   _            = torch.load(val_pt)

    # create DataLoaders
    train_ds = TensorDataset(train_imgs, train_lbls)
    val_ds   = TensorDataset(val_imgs, val_lbls)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=128)

    # init model
    model = AdvancedEmotionCNN()
    print("[Info] Created advanced CNN with first conv kernel=5, second=3, etc.\n")

    # train
    epochs = 10
    lr     = 0.001
    print(f"[Info] Training for {epochs} epochs, LR={lr} (Adam).")
    history = train_model(epochs, lr, model, train_dl, val_dl, optimizer_func=torch.optim.Adam)

    # plot
    plot_validation_accuracy(history)
    plot_train_vs_val_loss(history)

    # save
    os.makedirs("results", exist_ok=True)
    final_model_path = "results/advanced_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"[Info] Saved final model to '{final_model_path}'")

    # final train/val accuracy
    print("\n[Info] Evaluating entire training/validation sets...")
    train_acc = compute_full_dataset_accuracy(model, train_imgs, train_lbls)
    print(f"Train Set -> Accuracy: {train_acc * 100:.2f}%")

    val_acc = compute_full_dataset_accuracy(model, val_imgs, val_lbls)
    print(f"Val   Set -> Accuracy: {val_acc * 100:.2f}%")

    if os.path.exists(test_pt):
        print("[Info] test_data.pt found, can evaluate if desired. Done.")
    else:
        print("[Info] No test_data.pt found. Done.\n")


if __name__ == "__main__":
    main()
