"""
====================================================================================
FILE: download_dataset.py

PURPOSE:
  - This file downloads the "Face Expression Recognition Dataset" from Kaggle.
  - It uses the Kaggle API to authenticate and fetch the dataset, then unzips it
    into the local directory structure.

MAIN STEPS:
  1. Authenticate with Kaggle API.
  2. Specify the dataset to download ("jonathanoheix/face-expression-recognition-dataset").
  3. Create a local folder named 'data/face-expression-recognition-dataset/'.
  4. Download and unzip the dataset into that folder.

EXPECTED RESULT:
  - The folder 'data/face-expression-recognition-dataset/' should contain
    subfolders like 'images', and inside 'images' you should see 'train', 'validation',
    and possibly 'test'.

IMPLEMENTATION NOTES:
  - Kaggle credentials (kaggle.json) must be set up in ~/.kaggle/ or environment variables.
  - If you already have the dataset locally, you can skip this file.
====================================================================================
"""

import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset():
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

    # Create local output directory if it doesn't exist
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
