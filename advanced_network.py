"""
advanced_network.py

PURPOSE:
  - Demonstrates an advanced Convolutional Neural Network (CNN) architecture
    for 7-class facial expression recognition using PyTorch.
  - The architecture includes multiple Convolution + BatchNorm + ReLU + MaxPool + Dropout blocks,
    followed by flattening and two fully-connected layers with additional dropout.
  - Outputs standard training logs (epoch-based) and final accuracy on both
    the training and validation sets.
  - Saves final weights to 'results/advanced_model.pth' so that a similarly
    defined model in live_inference.py can be used for real-time classification.

REQUIREMENTS:
  - PyTorch, NumPy, Matplotlib, sklearn
  - 'processed_data/train_data.pt' and 'processed_data/val_data.pt' containing:
      (images_tensor, labels_tensor, label_to_idx).
    Each image: shape (N,1,48,48), each label: shape (N,).
    7 classes in total.

HOW TO RUN:
  python advanced_network.py

GENERAL EXPLANATIONS:
  1) This code is part of a project comparing several models:
       - Baseline (always predicts the largest class)
       - Just a Softmax (single linear layer)
       - Basic Fully Connected Network (MLP)
       - And this 'Advanced' CNN (uses multiple conv blocks with BN + Dropout)
  2) It is labeled "advanced" because it goes beyond simple flattening,
     extracting hierarchical features via convolutional layers (Conv2D),
     normalizing with BatchNorm, and applying pooling and dropout to reduce overfitting.
  3) The final model is typically more powerful for image tasks
     and is expected to outperform simpler baselines in facial expression recognition.
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
# Explanation: we are loading entire tuples from .pt, not just weights.
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
    :param batch_outputs: shape (batch_size, num_classes) - raw logits from the model
    :param batch_labels: shape (batch_size) - ground-truth class indices
    :return: torch scalar in [0..1] representing fraction of correct predictions

    Steps:
      1) Argmax across the 'num_classes' dimension => predicted label
      2) Compare to batch_labels to get the count of correct
      3) Divide by batch_size
    """
    _, predicted_indices = torch.max(batch_outputs, dim=1)
    correct_count = (predicted_indices == batch_labels).sum().item()
    accuracy_fraction = correct_count / len(batch_labels)
    return torch.tensor(accuracy_fraction)


class CNNTrainingBase(nn.Module):
    """
    A base class that defines standard methods for training and validation steps.
    This helps keep code organized. Each derived class can override or
    inherit these methods as needed.

    Main methods:
      - training_step(batch): compute forward pass + cross-entropy for training
      - validation_step(batch): returns dict of {'val_loss', 'val_acc'}
      - validation_epoch_end(batch_outputs): aggregates val_loss/val_acc across the epoch
      - epoch_end(epoch_idx, epoch_result): prints a summary line for each epoch
    """

    def training_step(self, batch):
        """
        One forward pass + cross-entropy for a training batch.
        The result is the training loss (a scalar).

        This will typically be called in a loop inside the training phase.
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

        Steps:
          1) forward pass
          2) compute cross-entropy loss
          3) compute accuracy fraction
          4) return them in a dictionary
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
        Returns a dict with final 'val_loss' and 'val_acc'.
        """
        losses = [x['val_loss'] for x in batch_outputs]
        val_loss_avg = torch.stack(losses).mean()
        accs   = [x['val_acc'] for x in batch_outputs]
        val_acc_avg  = torch.stack(accs).mean()
        return {'val_loss': val_loss_avg.item(), 'val_acc': val_acc_avg.item()}

    def epoch_end(self, epoch_idx, epoch_result):
        """
        Print a summary line for each epoch:
        showing train_loss, val_loss, val_acc
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
    The structure is intentionally more complex than a basic NN or a single-layer net:
      - 4 convolution blocks, each with Conv2D -> BN -> ReLU -> MaxPool -> Dropout
      - After flattening: 2 fully connected layers with BN + ReLU + Dropout
      - Final linear layer => 7-class logits

    This is 'advanced' compared to simpler MLP or single-layer models because
    it learns spatial features from the 48x48 input, capturing edges, corners,
    and more complex visual patterns that are crucial for distinguishing facial expressions.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Block 1: kernel=5 on first conv
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # Block 2: kernel=3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # Block 4
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # Flatten => e.g. shape becomes 512*3*3 = 4608
            nn.Flatten(),

            # Fully Connected #1 => (4608->256) + BN + ReLU + Dropout
            nn.Linear(512*3*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            # Fully Connected #2 => (256->512) + BN + ReLU + Dropout
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            # Final linear => 7 classes
            nn.Linear(512, 7)
        )

    def forward(self, x):
        """
        The forward pass simply applies self.network,
        which is a Sequential of all the conv blocks + flatten + FC layers.
        """
        return self.network(x)


##############################################################################
# 3) TRAIN/EVAL LOOPS
##############################################################################
@torch.no_grad()
def evaluate_model(model, data_loader):
    """
    Evaluate the model on the entire data_loader, computing average val_loss & val_acc.
    Returns a dict: {'val_loss', 'val_acc'}.

    Steps:
      - model.eval()
      - For each batch in data_loader, call model.validation_step()
      - Combine results with validation_epoch_end()
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
    Returns 'history': a list of dicts with keys {train_loss, val_loss, val_acc} per epoch.

    Flow per epoch:
      1) model.train()
      2) For each batch in train_loader => model.training_step() => accumulate loss => backward => step
      3) Evaluate on val_loader => evaluate_model(...)
      4) model.epoch_end(...) => logs the results
    """
    history = []
    optimizer = optimizer_func(model.parameters(), lr=learning_rate)

    for epoch_i in range(num_epochs):
        # 1) Training phase
        model.train()
        train_losses_this_epoch = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses_this_epoch.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 2) Validation phase
        val_results = evaluate_model(model, val_loader)

        # 3) Average train losses
        avg_train_loss = torch.stack(train_losses_this_epoch).mean().item()
        val_results['train_loss'] = avg_train_loss

        # 4) Print epoch summary
        model.epoch_end(epoch_i, val_results)
        history.append(val_results)

    return history

##############################################################################
# 4) PLOTTING
##############################################################################
def plot_validation_accuracy(history):
    """
    Plots the val_acc across epochs from 'history'.
    Each element in 'history' is a dict with {val_acc, val_loss, train_loss}.
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
    Plots training vs validation loss from the 'history'.
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
# 5) FULL-SET ACCURACY
##############################################################################
@torch.no_grad()
def compute_full_dataset_accuracy(model, images, labels):
    """
    Evaluate model's accuracy on an entire set at once (no mini-batching).
    images: shape (N,1,48,48)
    labels: shape (N,)
    returns a float in [0,1].
    """
    model.eval()
    outputs = model(images)  # shape (N,7) => raw logits
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total   = labels.size(0)
    return correct / total

##############################################################################
# 6) MAIN
##############################################################################
def main():
    """
    Main function steps:
      1) Check for processed train/val .pt files in 'processed_data/'.
      2) Load them, build DataLoaders.
      3) Initialize an AdvancedEmotionCNN (the 'advanced' approach).
      4) Train for a given number of epochs, track train_loss, val_loss, val_acc.
      5) Plot results, save final model to 'results/advanced_model.pth'.
      6) Print final accuracy on train/val sets, optionally on test if present.

    This is considered 'advanced' because it uses multiple convolution blocks
    to capture spatial features from the 48x48 grayscale images, significantly
    improving performance over simpler models like a single-layer or MLP.
    """
    train_pt = "processed_data/train_data.pt"
    val_pt   = "processed_data/val_data.pt"
    test_pt  = "processed_data/test_data.pt"

    # Check existence
    if not (os.path.exists(train_pt) and os.path.exists(val_pt)):
        print("[Error] Missing train_data.pt or val_data.pt!")
        return

    # 1) Load .pt files
    print("[Info] Loading training & validation data from .pt ...")
    train_imgs, train_lbls, label_to_idx = torch.load(train_pt)
    val_imgs,   val_lbls,   _            = torch.load(val_pt)

    # 2) Build Datasets & Loaders
    train_ds = TensorDataset(train_imgs, train_lbls)
    val_ds   = TensorDataset(val_imgs,  val_lbls)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=128)

    # 3) Create advanced CNN model
    model = AdvancedEmotionCNN()
    print("[Info] Created advanced CNN with multiple conv blocks.\n")
    print("[Note] This architecture is typically more powerful for images than a basic MLP.\n")

    # 4) Train
    epochs = 10
    lr     = 0.001
    print(f"[Info] Training for {epochs} epochs, LR={lr}, using Adam optimizer.\n")
    history = train_model(epochs, lr, model, train_dl, val_dl, optimizer_func=torch.optim.Adam)

    # 5) Plot
    plot_validation_accuracy(history)
    plot_train_vs_val_loss(history)

    # Save final model
    os.makedirs("results", exist_ok=True)
    final_model_path = "results/advanced_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"[Info] Saved final model to '{final_model_path}'\n")

    print("[Info] Checking final accuracies on entire train/val sets...")

    # Evaluate entire train set
    train_acc = compute_full_dataset_accuracy(model, train_imgs, train_lbls)
    print(f"Train Set -> Accuracy: {train_acc*100:.2f}%")

    # Evaluate entire val set
    val_acc = compute_full_dataset_accuracy(model, val_imgs, val_lbls)
    print(f"Val   Set -> Accuracy: {val_acc*100:.2f}%")

    # If test set exists, optionally evaluate
    if os.path.exists(test_pt):
        print("[Info] Found test_data.pt; user can evaluate similarly if needed.")
    else:
        print("[Info] No test_data.pt found. Done.\n")


if __name__ == "__main__":
    main()
