"""
========================================================================================================
FILE: dataset_preparation.py

PURPOSE:
  - This file prepares (preprocesses) the facial expression images for model training.
  - It converts images into grayscale, resizes them to 48x48 pixels, normalizes them,
    and saves them in .pt files for quick loading.

MAIN STEPS:
  1. Define paths for the train, validation, and test folders.
  2. Read images from each subfolder (where each subfolder is an emotion class).
  3. Apply transformations: Grayscale, Resize(48,48), ToTensor, Normalize(...).
  4. Save the processed tensors (images, labels, label_to_idx) as .pt files.

EXPECTED RESULT:
  - You should end up with 'processed_data/train_data.pt', 'processed_data/val_data.pt',
    and 'processed_data/test_data.pt'.

HOW IT FITS IN THE PROJECT:
  - The .pt files generated here will be used by baseline.py, softmax.py, basic_nn.py,
    and advanced_network.py for training and evaluation.

EXAMPLE USAGE:
  - Run this script once after preparing your dataset:
    python dataset_preparation.py

IMPLEMENTATION NOTES:
  - We use PyTorch transforms to standardize the images (grayscale, resizing, normalization).
  - All logs are printed to the console.
========================================================================================================
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

def collect_image_paths_and_labels(root_dir):
    """
    -------------------------------------------------------------------------------------
    FUNCTION: collect_image_paths_and_labels

    PURPOSE:
      - Recursively gather image file paths and their corresponding class labels
        (based on the folder structure).

    INPUT:
      root_dir (str): Path to a directory containing subfolders for each emotion/class.

    OUTPUT:
      data (list): A list of tuples (image_path, label_index).
      label_to_idx (dict): Mapping from label name (e.g. 'angry') to integer index (0..6).

    NOTES:
      - Each subfolder inside 'root_dir' is treated as one class (e.g. 'angry','happy' etc.).
      - Only .png, .jpg, .jpeg files are considered.
      - Raises FileNotFoundError if root_dir not found.
    -------------------------------------------------------------------------------------
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"[Error] Directory not found or not accessible: {root_dir}")

    classes = sorted(os.listdir(root_dir))
    label_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    data = []
    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            continue  # skip non-directory files

        for file_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file_name)
            if os.path.isfile(img_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append((img_path, label_to_idx[cls]))

    return data, label_to_idx

def load_and_preprocess_image(img_path, transform):
    """
    -------------------------------------------------------------------------------------
    FUNCTION: load_and_preprocess_image

    PURPOSE:
      - Opens an image file, applies the transform pipeline (e.g., resize, grayscale),
        and returns the resulting tensor.

    INPUT:
      img_path (str): Full path to an image file.
      transform (torchvision.transforms.Compose): Transform pipeline.

    OUTPUT:
      torch.Tensor (or None if corrupt).

    NOTES:
      - If an error occurs (corrupted file etc.), print warning and return None.
    -------------------------------------------------------------------------------------
    """
    try:
        with Image.open(img_path) as img:
            return transform(img)
    except Exception as e:
        print(f"[Warning] Could not process image: {img_path}. Error: {e}")
        return None

def prepare_dataset(dataset_root, output_path):
    """
    -------------------------------------------------------------------------------------
    FUNCTION: prepare_dataset

    PURPOSE:
      - Gathers images from `dataset_root`, applies transformations, saves to `output_path`.

    INPUT:
      dataset_root (str): Root directory containing subfolders for each class.
      output_path  (str): Path to save the .pt file.

    OUTPUT:
      label_to_idx (dict): class->index mapping.

    NOTES:
      - Prints how many images found/saved.
      - The .pt file contains (images_tensor, labels_tensor, label_to_idx).
    -------------------------------------------------------------------------------------
    """
    print("\n----------------------------------------------------")
    print(f"[INFO] Preparing dataset from: {dataset_root}")
    print("----------------------------------------------------")

    # transformation pipeline
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data, label_to_idx = collect_image_paths_and_labels(dataset_root)
    total_images = len(data)
    print(f"Found {total_images} images under {dataset_root}.\n")

    images = []
    labels = []

    for img_path, label_idx in tqdm(data, desc=f"Processing {os.path.basename(dataset_root)}"):
        tensor = load_and_preprocess_image(img_path, transform_pipeline)
        if tensor is not None:
            images.append(tensor)
            labels.append(label_idx)

    if len(images) == 0:
        raise ValueError(f"[Error] No valid images found in '{dataset_root}' after preprocessing.")

    images_tensor = torch.stack(images)                 # shape: (N,1,48,48)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Save the tuple
    torch.save((images_tensor, labels_tensor, label_to_idx), output_path)
    print(f"✓ Saved processed data for '{dataset_root}' -> {output_path}")

    return label_to_idx

def main():
    """
    ================================================================================
    FUNCTION: main (entry point)

    PURPOSE:
      - Defines folder paths for train, validation, test.
      - Prepares .pt files for each (train_data.pt, val_data.pt, test_data.pt).
      - Fails if the train/val/test directories don't exist.

    EXPECTED OUTPUT:
      - 'processed_data/train_data.pt', 'processed_data/val_data.pt',
        'processed_data/test_data.pt' all created if successful.

    STEPS:
      1) Ensure 'processed_data/' exists or create it.
      2) For each directory (train, val, test):
         - pass to `prepare_dataset(...)` -> *.pt
      3) Print final messages.
    ================================================================================
    """
    print("====================================================================")
    print("        DATASET PREPARATION FOR FACIAL EXPRESSION RECOGNITION       ")
    print("====================================================================\n")

    # Define your data structure
    train_dir = "data/face-expression-recognition-dataset/images/train"
    val_dir   = "data/face-expression-recognition-dataset/images/validation"
    test_dir  = "data/face-expression-recognition-dataset/images/test"  # now required

    # Output .pt paths
    os.makedirs("processed_data", exist_ok=True)
    train_out = "processed_data/train_data.pt"
    val_out   = "processed_data/val_data.pt"
    test_out  = "processed_data/test_data.pt"

    # 1) Prepare train set
    try:
        prepare_dataset(train_dir, train_out)
    except Exception as e:
        print(f"[Error] Training set preparation failed: {e}")
        sys.exit(1)

    # 2) Prepare validation set
    try:
        prepare_dataset(val_dir, val_out)
    except Exception as e:
        print(f"[Error] Validation set preparation failed: {e}")
        sys.exit(1)

    # 3) Prepare test set (now mandatory)
    if not os.path.isdir(test_dir):
        print(f"[Error] The test directory '{test_dir}' does not exist or is inaccessible.")
        sys.exit(1)

    try:
        prepare_dataset(test_dir, test_out)
    except Exception as e:
        print(f"[Error] Test set preparation failed: {e}")
        sys.exit(1)

    print("\n[Info] Dataset preparation completed successfully.")
    print("You should now have train_data.pt, val_data.pt, and test_data.pt in 'processed_data' folder.\n")
    print("Next steps:")
    print("  1) python baseline.py      (for baseline results)")
    print("  2) python softmax.py       (for Softmax classification)")
    print("  3) python basic_nn.py      (for basic fully-connected NN)")
    print("  4) python advanced_network.py (for advanced CNN model)")

if __name__ == "__main__":
    main()
