"""
======================================================================================================
FILE: models_comparison.py

PURPOSE:
  - Reads the final results from:
       results/baseline_results.txt
       results/softmax_results.txt
       results/basic_nn_results.txt
       results/advanced_network_results.txt
    Then displays them in a table and a bar chart for direct comparison.

HOW TO USE:
  1) Ensure you've run (in this order):
       - python dataset_preparation.py
       - python baseline.py
       - python softmax.py
       - python basic_nn.py
       - python advanced_network.py
     so that you have .txt files in 'results/'.

  2) Then run:
       - python models_comparison.py

OUTPUT:
  - A table summarizing final accuracies (baseline, softmax, basic NN, advanced CNN).
  - A bar chart to visualize them.
  - In-depth commentary on what these results mean.

REQUIREMENTS:
  - If you want pretty tables, install 'tabulate' (pip install tabulate).
  - Matplotlib for plotting bar charts.

======================================================================================================
"""

import os
import re
import matplotlib.pyplot as plt

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

def extract_accuracy_from_file(file_path):
    """
    Reads a results text file from 'results/' folder.
    Looks for lines like:
      - "Final Accuracy: <float>"
      - "Validation Accuracy: <float>"
    Returns that float or None if not found.
    """
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        content = f.read()

    # Regex for lines e.g. "Final Accuracy: 0.4367" / "Validation Accuracy: 0.5560"
    match = re.search(r"(?:Final Accuracy|Validation Accuracy):\s*([0-9.]+)", content)
    if match:
        return float(match.group(1))
    return None

def main():
    print("===========================================================================================")
    print("        COMPARATIVE ANALYSIS: BASELINE vs SOFTMAX vs BASIC NN vs ADVANCED CNN              ")
    print("===========================================================================================\n")

    # Path to results folder
    results_dir = "results"

    # Check if results folder even exists
    if not os.path.exists(results_dir):
        print("[Error] 'results' folder does not exist. Make sure you have run the training scripts first.")
        return

    # Filenames inside results folder
    baseline_file    = os.path.join(results_dir, "baseline_results.txt")
    softmax_file     = os.path.join(results_dir, "softmax_results.txt")
    basic_nn_file    = os.path.join(results_dir, "basic_nn_results.txt")
    advanced_cnn_file= os.path.join(results_dir, "advanced_network_results.txt")

    # Extract accuracies
    baseline_acc     = extract_accuracy_from_file(baseline_file)
    softmax_acc      = extract_accuracy_from_file(softmax_file)
    basic_nn_acc     = extract_accuracy_from_file(basic_nn_file)
    advanced_cnn_acc = extract_accuracy_from_file(advanced_cnn_file)

    # Prepare data
    results = [
        ("Baseline",       baseline_acc),
        ("Softmax",        softmax_acc),
        ("Basic NN (MLP)", basic_nn_acc),
        ("Advanced CNN",   advanced_cnn_acc)
    ]

    print("SUMMARY TABLE OF ACCURACIES:")
    table_data = []
    for name, acc in results:
        if acc is not None:
            table_data.append([name, f"{acc*100:.2f}%"])
        else:
            table_data.append([name, "No results found"])

    if TABULATE_AVAILABLE:
        print(tabulate(table_data, headers=["Model", "Final Accuracy"], tablefmt="fancy_grid"))
    else:
        print(" Model             | Final Accuracy")
        print("-------------------|----------------")
        for row in table_data:
            print(f"{row[0]:<19} | {row[1]}")

    # Make bar chart
    print("\nNow we create a bar chart for visual comparison...\n")

    valid_names      = [r[0] for r in results if r[1] is not None]
    valid_accuracies = [r[1] for r in results if r[1] is not None]

    if len(valid_accuracies) == 0:
        print("[Warning] No accuracies found at all. Maybe no scripts were run or no results exist.")
        return

    plt.figure(figsize=(8,5))
    plt.bar(valid_names, [a*100 for a in valid_accuracies], color=['gray','green','blue','purple'])
    plt.title("Comparison of Model Accuracies")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.show()

    # Commentary
    print("-------------------------------------------------------------------------------------------")
    print("IN-DEPTH COMMENTARY:")
    print("-------------------------------------------------------------------------------------------")
    print("1) **Baseline**: Minimal approach, typically the lowest accuracy. If your baseline is ~25%,")
    print("   any improvement above that means your model is learning some real patterns.\n")
    print("2) **Softmax** : Adds a linear classification layer over flattened images. If you see ~30-40%,")
    print("   it's a step up from baseline. Good, but might still be underfitting, as it doesn't capture")
    print("   spatial relations well.\n")
    print("3) **Basic NN**: A hidden layer (MLP) to capture non-linearities. Typically yields more gains,")
    print("   but watch out for overfitting if training loss is much lower than validation loss.\n")
    print("4) **Advanced CNN**: Convolutional layers exploit local features. Often best for image tasks.")
    print("   If your CNN outperforms the other methods significantly, it's expected. But if the gap is")
    print("   small, consider more epochs, data augmentation, or a deeper architecture.\n")

    print("=== Are these results good? ===")
    print(" - If your advanced CNN is around 50-60% or more, it's a decent start. Higher means even better.")
    print(" - If all models are very close to baseline, possibly there's an issue in data or training.\n")

    print("=== Over/Underfitting ===")
    print(" - Overfitting: If train loss is very low but val loss is high => the model memorizes training.")
    print(" - Underfitting: If both train and val losses remain high => the model isn't complex enough or ")
    print("   not trained enough.\n")

    print("=== Next Steps ===")
    print(" - Try more advanced data augmentation (flips, rotations).")
    print(" - Tweak hyperparameters (learning rates, batch sizes).")
    print(" - Add dropout or weight decay to reduce overfitting.")
    print(" - Use a separate test set if you want a final, unbiased estimate.\n")

    print("**Comparison completed.**")
    print("===========================================================================================\n")

if __name__ == "__main__":
    main()
