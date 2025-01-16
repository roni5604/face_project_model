"""
=====================================================================================
FILE: baseline.py

PURPOSE:
  - Implements a 'Baseline Model' which always predicts the most frequent class in the
    training set. This approach sets a minimum performance reference (around the proportion
    of the largest class).
  - Compares the actual performance metrics (accuracy, precision, recall) against more
    sophisticated models (Softmax, Basic NN, Advanced CNN) in the project.

HOW IT WORKS:
  1) Loads train_data.pt and val_data.pt from 'processed_data/'.
  2) Identifies the most frequent class label in the training labels.
  3) Predicts that single label for every example in the validation set.
  4) Calculates accuracy, precision, and recall for each class (most classes get zero).
  5) checks if test_data.pt exists, and evaluates the same baseline approach on it.


USAGE:
  python baseline.py

REQUIREMENTS:
  - Torch, NumPy, Matplotlib, sklearn
  - Processed data files (train_data.pt, val_data.pt, optional test_data.pt) in 'processed_data/'.

=====================================================================================
"""

import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

###############################################################################
# Utility function: get_class_distribution
###############################################################################
def get_class_distribution(labels):
    """
    Receives a 1D tensor 'labels' of size (N,) with integer class IDs.
    Returns a dictionary mapping:
      label_index -> count of how many times it appears.
    Example usage to see how many images belong to each label (class).
    """
    unique, counts = torch.unique(labels, return_counts=True)
    # Convert from torch tensors to regular Python int
    return {int(k.item()): int(v.item()) for k,v in zip(unique, counts)}

###############################################################################
# Main function
###############################################################################
def main():
    """
    main() steps:
      1) Print banner for "BASELINE MODEL: ALWAYS PREDICT MOST COMMON".
      2) Check for existence of train_data.pt and val_data.pt in 'processed_data/'.
      3) Load them, extract (train_images, train_labels), (val_images, val_labels), plus label_to_idx.
      4) Find the label with maximum count in train_labels.
      5) Predict that label for all val examples, compute accuracy, precision, recall, etc.
      6) Plot distribution of training classes, finalize baseline summary.
      7) If test_data.pt exists, do the same approach for the test set.
    """
    print("=======================================================")
    print("        BASELINE MODEL: ALWAYS PREDICT MOST COMMON     ")
    print("=======================================================\n")

    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)

    # Expected .pt file paths
    train_data_path = "processed_data/train_data.pt"
    val_data_path   = "processed_data/val_data.pt"
    test_data_path  = "processed_data/test_data.pt"  # Optional, if it exists

    # Check existence
    if not os.path.exists(train_data_path):
        print("[Error] Training data file not found. Please run dataset_preparation.py first.")
        return
    if not os.path.exists(val_data_path):
        print("[Error] Validation data file not found. Please run dataset_preparation.py first.")
        return

    # 1) Load the training & validation sets
    # Each .pt file typically has (images_tensor, labels_tensor, label_to_idx)
    train_images, train_labels, label_to_idx = torch.load(train_data_path, weights_only=False)
    val_images,   val_labels,   _            = torch.load(val_data_path,   weights_only=False)

    print("[Info] Successfully loaded training and validation data.")
    print(f" - Training set size: {train_labels.size(0)} images")
    print(f" - Validation set size: {val_labels.size(0)} images\n")

    # 2) Compute distribution in the training set
    #    e.g. label -> number_of_images
    train_distribution = get_class_distribution(train_labels)
    print("------------------------------------------------")
    print(" Class Distribution in Training Set (by index)  ")
    print("------------------------------------------------")
    for label_idx, count in train_distribution.items():
        # label_to_idx is a dict: class_name -> idx
        # we find the class_name that maps to this label_idx
        class_name = list(label_to_idx.keys())[list(label_to_idx.values()).index(label_idx)]
        print(f"  - {class_name} (index {label_idx}): {count} samples")

    # 3) Identify the most frequent class from training distribution
    most_frequent_class_idx = max(train_distribution, key=train_distribution.get)
    most_frequent_class_name = list(label_to_idx.keys())[list(label_to_idx.values()).index(most_frequent_class_idx)]

    print(f"\n[Info] The most frequent class is '{most_frequent_class_name}' (index {most_frequent_class_idx}).")
    print("       We will predict this class for every image in the validation set.\n")

    # 4) Evaluate baseline on validation data
    val_predictions = np.full_like(val_labels.numpy(), most_frequent_class_idx)
    val_true        = val_labels.numpy()

    accuracy  = accuracy_score(val_true, val_predictions)
    precision = precision_score(val_true, val_predictions, average=None, zero_division=0)
    recall    = recall_score(val_true, val_predictions, average=None, zero_division=0)

    print("=========================================")
    print("         BASELINE MODEL METRICS          ")
    print("=========================================\n")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

    print("Precision and Recall for each class:")
    for idx, (p, r) in enumerate(zip(precision, recall)):
        class_name = list(label_to_idx.keys())[list(label_to_idx.values()).index(idx)]
        print(f"  - {class_name:8s} -> Precision: {p:.3f}, Recall: {r:.3f}")

    # 5) Plot the distribution of training classes as a bar chart
    labels_sorted = sorted(train_distribution.keys())
    counts_sorted = [train_distribution[k] for k in labels_sorted]

    plt.figure(figsize=(8,5))
    plt.bar(range(len(labels_sorted)), counts_sorted, color='skyblue')
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Class Label Index")
    plt.ylabel("Number of Samples")
    x_tick_names = [list(label_to_idx.keys())[list(label_to_idx.values()).index(k)] for k in labels_sorted]
    plt.xticks(range(len(labels_sorted)), x_tick_names, rotation=45)
    plt.tight_layout()
    plt.show()

    print("\n[Info] Baseline evaluation completed. This sets a 'minimum' performance to beat.\n")

    # 6) Save final baseline accuracy to a file
    baseline_file = "results/baseline_results.txt"
    with open(baseline_file, "w") as f:
        f.write(f"Final Accuracy: {accuracy:.4f}\n")
    print(f"[Info] Baseline final accuracy saved to '{baseline_file}'.\n")

    # -------------------------------------------------------
    # ADDITIONAL SUMMARY: Evaluate baseline on train, val, test (if test exists)
    # -------------------------------------------------------
    print("-------------------------------------------------------")
    print("ADDITIONAL SUMMARY: BASELINE ON TRAIN, VAL, AND TEST")
    print("-------------------------------------------------------")

    def evaluate_baseline(labels_tensor, freq_idx):
        """
        Given a labels tensor (e.g. train_labels or val_labels),
        generate a prediction array that always picks 'freq_idx'.
        Then compute accuracy, precision, recall.
        Returns (acc, prec, rec) to allow further inspection.
        """
        y_pred = np.full_like(labels_tensor.numpy(), freq_idx)
        y_true = labels_tensor.numpy()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
        return acc, prec, rec

    # Evaluate on the entire train set with the baseline approach
    train_acc, train_prec, train_rec = evaluate_baseline(train_labels, most_frequent_class_idx)
    print(f"Train Set -> Accuracy: {train_acc*100:.2f}%")

    # Evaluate on the entire val set (which we already did, but for completeness)
    val_acc, val_prec, val_rec = evaluate_baseline(val_labels, most_frequent_class_idx)
    print(f"Val   Set -> Accuracy: {val_acc*100:.2f}%")

    # Evaluate on test set if it exists
    if os.path.exists(test_data_path):
        test_images, test_labels, _ = torch.load(test_data_path, weights_only=False)
        test_acc, _, _ = evaluate_baseline(test_labels, most_frequent_class_idx)
        print(f"Test  Set -> Accuracy: {test_acc*100:.2f}%")
    else:
        print("[Info] No test set found, skipping test metrics.")

    print("Precision and Recall metrics for each set are computed similarly. You can adapt them if needed.")
    print("End of additional summary.\n")


if __name__ == "__main__":
    main()
