"""
=====================================================================================
FILE: download_dataset.py

PURPOSE:
  - Download the "Face Expression Recognition Dataset" (or a similar Kaggle dataset)
    from Kaggle using the Kaggle API.
  - Unzip it into a local directory so that other scripts (dataset_preparation.py)
    can then read and prepare the data.

PREREQUISITES:
  - Kaggle credentials (kaggle.json) properly configured in ~/.kaggle/.
  - 'kaggle' Python package installed (pip install kaggle).

HOW TO RUN:
  1) python download_dataset.py
  2) Check data/face-expression-recognition-dataset/ for downloaded files.

EXPECTED OUTCOME:
  - A directory "data/face-expression-recognition-dataset/" with images subfolders
    like 'train/', 'validation/', 'test/' (depending on the dataset structure).

=====================================================================================
"""

import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    """
    Steps:
      1) Print a banner about the dataset download.
      2) Authenticate with Kaggle API.
      3) Download and unzip the dataset "jonathanoheix/face-expression-recognition-dataset".
      4) Place it inside 'data/face-expression-recognition-dataset/'.
      5) Print logs about success or any errors.
    """
    print("========================================================================")
    print("    DOWNLOADING FACIAL EXPRESSION RECOGNITION DATASET FROM KAGGLE      ")
    print("========================================================================\n")

    try:
        print("Attempting to authenticate with Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle API authentication successful.")
    except Exception as e:
        print("\n[Error] Could not authenticate with Kaggle API. Please check your Kaggle credentials.")
        raise e

    dataset = "jonathanoheix/face-expression-recognition-dataset"
    output_dir = "data/face-expression-recognition-dataset/"
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"\nDownloading dataset '{dataset}' from Kaggle...")
        api.dataset_download_files(dataset, path=output_dir, unzip=True)
        print(f"✓ Dataset downloaded and unzipped successfully into: {output_dir}")
    except Exception as e:
        print("\n[Error] Could not download or unzip the dataset.")
        raise e

    print("\n[Info] Download completed. You can now proceed to 'dataset_preparation.py' for preprocessing.\n")


if __name__ == "__main__":
    download_dataset()
