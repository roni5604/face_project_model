"""
========================================================================================================
FILE: dataset_preparation.py

PURPOSE:
  - This file prepares (preprocesses) the facial expression images for model training.
  - It converts images into grayscale, resizes them to 48x48 pixels, normalizes them,
    and saves them in .pt files for quick loading.

MAIN STEPS:
  1. Define paths for the train, validation, and (optional) test folders.
  2. Read images from each subfolder (where each subfolder is an emotion class).
  3. Apply transformations: Grayscale, Resize(48,48), ToTensor, Normalize(...).
  4. Save the processed tensors (images, labels, label_to_idx) as .pt files.

EXPECTED RESULT:
  - You should end up with 'processed_data/train_data.pt', 'processed_data/val_data.pt',
    and (if available) 'processed_data/test_data.pt'.

HOW IT FITS IN THE PROJECT:
  - The .pt files generated here will be used by the baseline, softmax, basic_nn,
    and advanced_network scripts for training and evaluation.

EXAMPLE USAGE:
  - Run this script once after downloading the dataset:
    python dataset_preparation.py

IMPLEMENTATION NOTES:
  - We use PyTorch transforms to standardize the images.
  - We do not log to a file; all info is printed to the console.
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
      label_to_idx (dict): Mapping from label (e.g. 'angry') to integer index (e.g. 0).

    NOTES:
      - Each subfolder inside 'root_dir' is treated as a unique class.
      - We only consider files with .png, .jpg, or .jpeg extensions.
    -------------------------------------------------------------------------------------
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"[Error] The directory {root_dir} does not exist or is not accessible.")

    classes = sorted(os.listdir(root_dir))
    label_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    data = []
    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            # If it's not a folder, skip
            continue

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
      - Opens an image file, applies a given transformation pipeline (e.g. resize),
        and returns the resulting tensor.

    INPUT:
      img_path (str): Full path to an image file.
      transform (torchvision.transforms.Compose): Transform pipeline to apply.

    OUTPUT:
      torch.Tensor (or None if the image is corrupted).

    NOTES:
      - If any error occurs (e.g., corrupted image), we print a warning and return None.
    -------------------------------------------------------------------------------------
    """
    try:
        with Image.open(img_path) as img:
            img_tensor = transform(img)
        return img_tensor
    except Exception as e:
        print(f"[Warning] Could not process image: {img_path}. Error: {e}")
        return None


def prepare_dataset(dataset_root, output_path):
    """
    -------------------------------------------------------------------------------------
    FUNCTION: prepare_dataset

    PURPOSE:
      - Gathers all images in `dataset_root`, applies transformations (grayscale, resize,
        etc.), and saves the resulting tensors to `output_path`.

    INPUT:
      dataset_root (str): Root directory containing subfolders for each class.
      output_path  (str): Path to save the processed dataset .pt file.

    OUTPUT:
      label_to_idx (dict): The class-to-index mapping used.

    NOTES:
      - We print out how many images were found and saved.
      - The .pt file saves (images_tensor, labels_tensor, label_to_idx).
    -------------------------------------------------------------------------------------
    """
    print("\n----------------------------------------------------")
    print(f"[INFO] Preparing dataset from: {dataset_root}")
    print("----------------------------------------------------")

    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Collect image paths and labels
    data, label_to_idx = collect_image_paths_and_labels(dataset_root)
    total_images = len(data)
    print(f"Found {total_images} images under {dataset_root}.\n")

    images = []
    labels = []

    for img_path, label in tqdm(data, desc=f"Processing {os.path.basename(dataset_root)}"):
        tensor = load_and_preprocess_image(img_path, transform_pipeline)
        if tensor is not None:
            images.append(tensor)
            labels.append(label)

    if len(images) == 0:
        raise ValueError(f"[Error] No valid images found in {dataset_root} after preprocessing.")

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Save to .pt
    torch.save((images_tensor, labels_tensor, label_to_idx), output_path)
    print(f"✓ Saved processed data for '{dataset_root}' -> {output_path}")

    return label_to_idx


def main():
    """
    ================================================================================
    FUNCTION: main (entry point)

    PURPOSE:
      - Defines the specific folder paths for train, validation, test (optional)
        then calls `prepare_dataset()` to create the .pt files.
      - Prints progress and results directly to the console.

    EXPECTED OUTPUT:
      - The files train_data.pt, val_data.pt, and optionally test_data.pt
        in the `processed_data` folder.

    STEPS:
      1. Create 'processed_data' folder if it doesn't exist.
      2. Prepare train set -> processed_data/train_data.pt
      3. Prepare validation set -> processed_data/val_data.pt
      4. (Optional) Prepare test set -> processed_data/test_data.pt
      5. Print messages with results.
    ================================================================================
    """
    print("====================================================================")
    print("               DATASET PREPARATION FOR FACIAL EXPRESSIONS           ")
    print("====================================================================\n")

    # Folder structure for your dataset
    train_dir = "data/face-expression-recognition-dataset/images/train"
    val_dir = "data/face-expression-recognition-dataset/images/validation"
    test_dir = "data/face-expression-recognition-dataset/images/test"  # optional

    # Output .pt paths
    os.makedirs("processed_data", exist_ok=True)
    train_out = "processed_data/train_data.pt"
    val_out = "processed_data/val_data.pt"
    test_out = "processed_data/test_data.pt"

    # Prepare train set
    try:
        prepare_dataset(train_dir, train_out)
    except Exception as e:
        print(f"[Error] Training set preparation failed: {e}")
        sys.exit(1)

    # Prepare validation set
    try:
        prepare_dataset(val_dir, val_out)
    except Exception as e:
        print(f"[Error] Validation set preparation failed: {e}")
        sys.exit(1)

    # Prepare test set (if exists)
    if os.path.isdir(test_dir):
        try:
            prepare_dataset(test_dir, test_out)
        except Exception as e:
            print(f"[Error] Test set preparation failed: {e}")
    else:
        print("\n[Warning] No test directory found. Skipping test set creation.\n")

    print("\n[Info] Dataset preparation completed successfully.")
    print("You should now have .pt files in 'processed_data' folder.\n")
    print("Next steps:")
    print("  1) Run: python baseline.py       (to get baseline results)")
    print("  2) Run: python softmax.py        (to train a Softmax regression model)")
    print("  3) Run: python basic_nn.py       (to train a basic fully-connected NN)")
    print("  4) Run: python advanced_network.py (to train a CNN)\n")


if __name__ == "__main__":
    main()
