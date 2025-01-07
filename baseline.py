"""
========================================================================================================
FILE: baseline.py

PURPOSE:
  - Implements the simplest baseline model: always predict the most frequent class.
  - Evaluates on the validation set, prints metrics, and saves baseline_results.txt in 'results/'.

ADDED:
  - A second summary block that also evaluates train, val, (and test if exists) all together,
    printing accuracy, precision, recall.
========================================================================================================
"""

import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def get_class_distribution(labels):
    unique, counts = torch.unique(labels, return_counts=True)
    return {int(k.item()): int(v.item()) for k,v in zip(unique, counts)}

def main():
    print("=======================================================")
    print("        BASELINE MODEL: ALWAYS PREDICT MOST COMMON     ")
    print("=======================================================\n")

    os.makedirs("results", exist_ok=True)
    train_data_path = "processed_data/train_data.pt"
    val_data_path   = "processed_data/val_data.pt"
    test_data_path  = "processed_data/test_data.pt"  # Optional, if it exists

    if not os.path.exists(train_data_path):
        print("[Error] Training data file not found. Please run dataset_preparation.py first.")
        return
    if not os.path.exists(val_data_path):
        print("[Error] Validation data file not found. Please run dataset_preparation.py first.")
        return

    # Load data
    train_images, train_labels, label_to_idx = torch.load(train_data_path, weights_only=False)
    val_images,   val_labels,   _            = torch.load(val_data_path,   weights_only=False)

    print("[Info] Successfully loaded training and validation data.")
    print(f" - Training set size: {train_labels.size(0)} images")
    print(f" - Validation set size: {val_labels.size(0)} images\n")

    # Distribution
    train_distribution = get_class_distribution(train_labels)
    print("------------------------------------------------")
    print(" Class Distribution in Training Set (by index)  ")
    print("------------------------------------------------")
    for label_idx, count in train_distribution.items():
        class_name = list(label_to_idx.keys())[list(label_to_idx.values()).index(label_idx)]
        print(f"  - {class_name} (index {label_idx}): {count} samples")

    # Identify most frequent class
    most_frequent_class_idx = max(train_distribution, key=train_distribution.get)
    most_frequent_class_name = list(label_to_idx.keys())[list(label_to_idx.values()).index(most_frequent_class_idx)]

    print(f"\n[Info] The most frequent class is '{most_frequent_class_name}' (index {most_frequent_class_idx}).")
    print("       We will predict this class for every image in the validation set.\n")

    # Baseline predictions (val)
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

    # Plot distribution
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

    # Save final accuracy
    baseline_file = "results/baseline_results.txt"
    with open(baseline_file, "w") as f:
        f.write(f"Final Accuracy: {accuracy:.4f}\n")
    print(f"[Info] Baseline final accuracy saved to '{baseline_file}'.\n")

    # -------------------------------------------------------
    # ADDITIONAL SUMMARY BLOCK: Evaluate train, val, test (if exists)
    # -------------------------------------------------------
    print("-------------------------------------------------------")
    print("ADDITIONAL SUMMARY: BASELINE ON TRAIN, VAL, AND TEST")
    print("-------------------------------------------------------")

    # Evaluate function for baseline
    def evaluate_baseline(labels_tensor, freq_idx):
        # Predict freq_idx for all
        y_pred = np.full_like(labels_tensor.numpy(), freq_idx)
        y_true = labels_tensor.numpy()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec = recall_score(y_true, y_pred, average=None, zero_division=0)
        return acc, prec, rec

    # Train
    train_acc, train_prec, train_rec = evaluate_baseline(train_labels, most_frequent_class_idx)
    print(f"Train Set -> Accuracy: {train_acc*100:.2f}%")
    # Val
    val_acc, val_prec, val_rec = evaluate_baseline(val_labels, most_frequent_class_idx)
    print(f"Val   Set -> Accuracy: {val_acc*100:.2f}%")

    # Test (if exists)
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
