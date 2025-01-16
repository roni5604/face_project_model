
# **Deep Learning: Facial Expression Recognition Project**  
**By Roni Michaeli, Beni Tibi, and Amit Kabalo**  
**Ariel University**

---

## **Overview**

This repository showcases a **facial expression recognition** system using **deep learning**. The system compares **four** main approaches:

1. **Baseline Model**  
   - Always predicts the most frequent class (no learning).
2. **Softmax Model**  
   - A single-layer linear classifier (logistic regression for multiple classes).
3. **Basic Fully Connected Network (MLP)**  
   - A simple neural network flattening the image and using fully connected layers.
4. **Advanced Convolutional Neural Network (CNN)**  
   - Multiple convolutional blocks (Conv2D, BatchNorm, ReLU, MaxPool, Dropout) for superior performance.

Each model is trained to recognize **7 emotion classes** (angry, disgust, fear, happy, neutral, sad, surprise) from grayscale images sized **48×48**. After training, we demonstrate a **live inference** script where the advanced CNN classifies expressions in **real time** via a webcam feed.

---

## **Purpose**

1. **Educational Goal**  
   - Illustrate how different neural network architectures perform on the same dataset and conditions, highlighting the evolution from trivial baseline to advanced CNN.

2. **Comparison**  
   - Evaluate accuracy, precision, recall for each approach.  
   - Show that the **CNN** typically outperforms simpler models on an image-based problem.

3. **Practical Flow**  
   - **Data Download** (from Kaggle)  
   - **Data Preparation** (resize, grayscale, normalization)  
   - **Training** multiple models  
   - **Evaluation** (metrics)  
   - **Live Inference** using the advanced model.

---

## **Dataset**

- **Source**: [Kaggle - Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)  
- Contains 7 categories in subfolders: e.g., `train/angry/`, `train/happy/`, etc.
- Images are manually split into `train/`, `validation/`, and `test/` subfolders.

---

## **Installation Instructions**

1. **Clone the repository** (or download):
   ```bash
   git clone https://github.com/YourUserName/facial_expression_recognition_project.git
   cd facial_expression_recognition_project
   ```

2. **(Optional) Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```

3. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
   ```
   - This includes **PyTorch**, **NumPy**, **Matplotlib**, **OpenCV**, **scikit-learn**, **kaggle**, etc.

4. **Kaggle Setup** (if you need to download the dataset):
   - Place your `kaggle.json` credentials in `~/.kaggle/` (Linux/Mac) or `%HOMEPATH%\.kaggle\` (Windows).  
   - Ensure correct permissions (e.g., `chmod 600 ~/.kaggle/kaggle.json`).

5. **Download dataset** (if required):
   ```bash
   python download_dataset.py
   ```
   - This will fetch the Face Expression Recognition dataset and unzip it under `data/face-expression-recognition-dataset/`.

6. **Prepare the `.pt` data** (preprocessing):
   ```bash
   python dataset_preparation.py
   ```
   - Reads raw images from `train/`, `validation/`, `test/`, converts them to **(N,1,48,48)** shape, normalizes, and saves `.pt` files to `processed_data/`.

---

## **Project Structure**

```
facial_expression_recognition_project/
├─ data/
│   └─ face-expression-recognition-dataset/
│       └─ images/
│           ├─ train/
│           ├─ validation/
│           └─ test/
├─ processed_data/
│   ├─ train_data.pt
│   ├─ val_data.pt
│   └─ test_data.pt
├─ results/
│   ├─ baseline_results.txt
│   ├─ softmax_model.pth
│   ├─ basic_nn_results.txt
│   └─ advanced_model.pth
├─ download_dataset.py
├─ dataset_preparation.py
├─ baseline.py
├─ softmax.py
├─ basic_nn.py
├─ advanced_network.py
├─ live_inference.py
├─ requirements.txt
└─ readme.md
```

---

## **Code Explanations**

1. **`download_dataset.py`**  
   - Authenticates to Kaggle API and downloads the **Face Expression Recognition Dataset**, unzipping into `data/face-expression-recognition-dataset/`.  
   - If you already have the dataset, you can skip.

2. **`dataset_preparation.py`**  
   - Reads images from `train/`, `validation/`, `test/` subfolders.  
   - Applies **transforms**: grayscale, resizing to 48×48, converting to tensor, normalizing to ~[-1..1].  
   - Saves `train_data.pt`, `val_data.pt`, `test_data.pt` in `processed_data/`.  
   - These files are loaded by all other scripts, ensuring consistent input shapes and label mappings.

3. **`baseline.py`**  
   - Computes the **most frequent class** in the training set.  
   - Always predicts that class for every sample (train/val/test).  
   - Serves as a minimum reference performance.

4. **`softmax.py`**  
   - A **single-layer** logistic regression model with cross-entropy.  
   - Flattens each 48×48 image to a 2304-element vector → single linear layer → 7 outputs.  
   - Trains for a specified number of epochs, prints classification metrics.

5. **`basic_nn.py`**  
   - A **Fully Connected (MLP)** approach, flattening images but adding 2 hidden layers with ReLU, BN, and Dropout.  
   - Gains more nonlinearity over softmax, yet does not exploit spatial structure.

6. **`advanced_network.py`**  
   - **Advanced CNN** with multiple **Conv2D** blocks (conv + BN + ReLU + pool + dropout) followed by flatten and fully connected layers.  
   - Typically yields the best accuracy on image tasks.  
   - Saves model weights to `results/advanced_model.pth`.

7. **`live_inference.py`**  
   - Performs **real-time** emotion detection from a webcam feed using the advanced CNN.  
   - Detects faces with OpenCV’s Haar cascade, resizes to 48×48, passes them into the CNN, draws bounding boxes and labels on the video stream.

---

## **How to Run the Models**

1. **Baseline**  
   ```bash
   python baseline.py
   ```
   - Logs trivial approach accuracy, saves results to `results/baseline_results.txt`.

2. **Softmax**  
   ```bash
   python softmax.py
   ```
   - Trains the linear model, saves final state dict as `softmax_model.pth`.

3. **Basic NN (MLP)**  
   ```bash
   python basic_nn.py
   ```
   - Multi-layer perceptron, logs classification report, saves results in `basic_nn_results.txt`.

4. **Advanced CNN**  
   ```bash
   python advanced_network.py
   ```
   - Trains the deeper convolutional network, saves best weights to `advanced_model.pth`.

5. **Live Inference**  
   ```bash
   python live_inference.py
   ```
   - Must have `advanced_model.pth` from the advanced CNN.  
   - Opens webcam, detects face, classifies expression in real time. Press `q` to quit.

---

## **Contact / Credits**

- **Authors**:  
  - Roni Michaeli  
  - Beni Tibi  
  - Amit Kabalo  
- **Institution**: *Ariel University*  
- **Course**: *Deep Learning*

**Kaggle Reference**:  
[Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

---

## **Future Ideas**

1. **Data Augmentation** (random flips/rotations) to improve generalization.  
2. **Class Weights** or **oversampling** for underrepresented expressions (e.g., disgust).  
3. **Transfer Learning** from pretrained networks (like ResNet) for potentially higher accuracy.  
4. **Hyperparameter Tuning** (learning rate, dropout rates, layer dimensions).

---

## **Conclusion**

This **Deep Learning** project demonstrates the progression from a **baseline** approach to an **advanced CNN** for recognizing facial expressions from images. By comparing each model’s metrics, one can appreciate how convolutional layers significantly improve performance on image-based tasks. The **live inference** script further illustrates practical usage, showing real-time classification from a webcam feed.

**Thank you** for reviewing our project submission! We hope it clearly demonstrates the **end-to-end** pipeline of data preparation, model training, evaluation, and real-time inference.