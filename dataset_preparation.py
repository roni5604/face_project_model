"""
=============================================================================================
FILE: dataset_preparation.py

PURPOSE:
  - Prepares (preprocesses) the facial expression images for model training and evaluation.
  - Converts images (from Kaggle dataset) into grayscale, resizes to 48x48 pixels, normalizes,
    and saves them in .pt files for quick loading by PyTorch.

GENERAL EXPLANATIONS:
  1) The images come from a Kaggle dataset ("jonathanoheix/face-expression-recognition-dataset")
     that was downloaded and extracted using 'download_dataset.py'.
  2) We assume the dataset has 'train/', 'validation/', and 'test/' subfolders, each containing
     images of different emotion classes (angry, happy, etc.).
  3) We apply transformations:
       - Grayscale() => 1-channel image
       - Resize(48,48)
       - ToTensor() => shape (N,1,48,48)
       - Normalize((0.5,), (0.5,)) => scale pixel range to roughly [-1..1]
  4) Then we save the processed sets into:
       processed_data/train_data.pt
       processed_data/val_data.pt
       processed_data/test_data.pt
  5) These .pt files are used by all training scripts (baseline, softmax, basic_nn, advanced_network).

HOW IT FITS THE PROJECT:
  - This script is crucial for ensuring every model sees the same input format (N,1,48,48).
  - The advanced model (CNN) especially relies on consistent input shape for convolution layers.
  - Later, "live_inference.py" matches the same normalization so real-time images
    can be recognized consistently.

IMPLEMENTATION NOTES:
  - We log info about how many images were found in each subfolder and how many were successfully processed.
  - If no images exist or a folder is missing, we raise an error (and possibly stop).
=============================================================================================
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
        based on the folder structure (each subfolder = one emotion class).

    INPUT:
      root_dir (str): e.g., "data/face-expression-recognition-dataset/images/train".

    OUTPUT:
      data (list of (img_path, label_idx)).
      label_to_idx (dict): class_name -> integer index.

    NOTES:
      - We only consider files with .png, .jpg, .jpeg extensions.
      - Sorting subfolder names ensures consistent labeling across runs.
    -------------------------------------------------------------------------------------
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"[Error] The directory {root_dir} does not exist or is not accessible.")

    # Each subfolder inside root_dir is a class name
    classes = sorted(os.listdir(root_dir))
    # Create a mapping class_name -> integer index
    label_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    data = []
    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            # skip if it's not a subfolder
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
      - Opens an image file with PIL, applies the given 'transform' pipeline,
        and returns a PyTorch tensor.

    INPUT:
      img_path (str): path to image.
      transform (torchvision.transforms.Compose): pipeline of transformations.

    OUTPUT:
      torch.Tensor => shape (1,48,48) if successful, or None if error/corrupt image.

    NOTES:
      - If an Exception occurs (like corrupted file), we print a warning and return None.
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
      - Gathers images from `dataset_root`, applies transformations, and saves
        the final (images_tensor, labels_tensor, label_to_idx) to `output_path`.

    INPUT:
      dataset_root (str): e.g. "data/face-expression-recognition-dataset/images/train"
      output_path  (str): e.g. "processed_data/train_data.pt"

    OUTPUT:
      label_to_idx (dict): the mapping from class_name to integer index used here.

    NOTES:
      - Prints the number of images found and how many are successfully saved.
      - If zero images are processed, raises ValueError.
    -------------------------------------------------------------------------------------
    """
    print("\n----------------------------------------------------")
    print(f"[INFO] Preparing dataset from: {dataset_root}")
    print("----------------------------------------------------")

    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # scale to approx [-1..1]
    ])

    # collect all (img_path, label_idx)
    data, label_to_idx = collect_image_paths_and_labels(dataset_root)
    total_images = len(data)
    print(f"Found {total_images} images under {dataset_root}.\n")

    images = []
    labels = []

    # apply transformations for each image
    for img_path, label_idx in tqdm(data, desc=f"Processing {os.path.basename(dataset_root)}"):
        tensor = load_and_preprocess_image(img_path, transform_pipeline)
        if tensor is not None:
            images.append(tensor)
            labels.append(label_idx)

    if len(images) == 0:
        raise ValueError(f"[Error] No valid images found in '{dataset_root}' after preprocessing.")

    images_tensor = torch.stack(images)  # shape: (N,1,48,48)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # save as a tuple in .pt format
    torch.save((images_tensor, labels_tensor, label_to_idx), output_path)
    print(f"✓ Saved processed data for '{dataset_root}' -> {output_path}")

    return label_to_idx


def main():
    """
    ================================================================================
    FUNCTION: main (entry point)

    PURPOSE:
      - Defines folder paths for train, validation, test sets (under 'images/').
      - Calls prepare_dataset(...) on each, creating .pt files in 'processed_data/'.
      - If any step fails, we log the error and potentially exit.

    NOTES:
      - Once completed, you should have train_data.pt, val_data.pt, test_data.pt
        in 'processed_data/'.
      - These are essential for training baseline.py, softmax.py, basic_nn.py,
        advanced_network.py. Later, 'live_inference.py' is used for real-time testing
        of the advanced model.

    1) train_dir = "data/.../train"
    2) val_dir   = "data/.../validation"
    4) prepare_dataset for each => ".pt" files
    5) Print final info & next steps
    ================================================================================
    """
    print("====================================================================")
    print("        DATASET PREPARATION FOR FACIAL EXPRESSION RECOGNITION       ")
    print("====================================================================\n")

    # 1) define the subfolders for train/val/test
    train_dir = "data/face-expression-recognition-dataset/images/train"
    val_dir   = "data/face-expression-recognition-dataset/images/validation"

    # 2) create 'processed_data' folder
    os.makedirs("processed_data", exist_ok=True)

    # define output paths
    train_out = "processed_data/train_data.pt"
    val_out   = "processed_data/val_data.pt"

    # 3) prepare train set
    try:
        prepare_dataset(train_dir, train_out)
    except Exception as e:
        print(f"[Error] Training set preparation failed: {e}")
        sys.exit(1)

    # 4) prepare validation set
    try:
        prepare_dataset(val_dir, val_out)
    except Exception as e:
        print(f"[Error] Validation set preparation failed: {e}")
        sys.exit(1)


    print("\n[Info] Dataset preparation completed successfully.")
    print("You should now have train_data.pt, val_data.pt, test_data.pt in 'processed_data/'.\n")
    print("Next steps:")
    print("  1) python baseline.py      (baseline model)")
    print("  2) python softmax.py       (softmax linear model)")
    print("  3) python basic_nn.py      (basic fully-connected MLP)")
    print("  4) python advanced_network.py (advanced CNN)")
    print("\nFinally, after training the advanced model, you can run 'live_inference.py' to see")
    print("real-time detection of emotions via webcam feed, using the advanced CNN.")


if __name__ == "__main__":
    main()
