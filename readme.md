

## ** Installation, Running, and Output**

### **1. Overview**

This repository contains a **Facial Expression Recognition** pipeline that classifies images (or live webcam frames) into seven possible emotions:

```
1) Angry
2) Disgust
3) Fear
4) Happy
5) Neutral
6) Sad
7) Surprise
```

The project includes:

- **Baseline** model: Always predicts the most frequent class.  
- **Softmax/Logistic** model: Simple linear approach with cross-entropy.  
- **Basic NN** (MLP): A fully connected network (one or more hidden layers).  
- **Advanced CNN**: A convolutional network with multiple conv-pool blocks, dropout, and possibly weight decay.  
- **Live Inference**: A script that uses the best model in real time with a webcam to detect or classify facial expressions.

### **2. Requirements**

- **Python 3.9+** (tested; versions 3.7+ likely work)
- **PyTorch** (for building/training/inference)
- **NumPy**
- **OpenCV** (for live webcam usage)
- **Matplotlib** (for plotting)
- **Scikit-learn** (for classification metrics)

Example environment setup (Mac, Linux, or Windows):
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install --upgrade pip
pip install torch torchvision numpy opencv-python matplotlib scikit-learn
```

### **3. Files and What They Do**

```
root/
│
├─ baseline.py
│   - Implements a naive model that always predicts the most frequent class.
│
├─ softmax.py
│   - Trains a simple Softmax regression model on flattened 48x48 images.
│
├─ basic_nn.py
│   - Trains a fully connected (MLP) neural network with one or more hidden layers.
│
├─ advanced_network.py
│   - Trains a deeper CNN with multiple convolutional layers, dropout, early stopping, etc.
│   - Produces best results and saves model weights to results/advanced_model.pth
│
├─ live_inference.py
│   - Runs a live webcam session, detects face (optionally with Haar Cascade),
│     classifies it using the advanced CNN, draws bounding boxes, displays emotions.
│
├─ processed_data/ (folder)
│   - Contains .pt files (train_data.pt, val_data.pt, test_data.pt) with preprocessed images/labels
│
├─ results/ (folder)
│   - Stores output logs, final models (e.g., advanced_model.pth), classification reports, etc.
│
├─ readme.md (this file)
│
└─ etc...
```

### **4. How to Run**

**(A) Dataset Preparation**  
1. Download or collect your **facial expression dataset** (e.g., 48×48 grayscale).  
2. Use a script like `dataset_preparation.py` (if available) or your own approach to produce `.pt` files:
   ```
   processed_data/train_data.pt
   processed_data/val_data.pt
   processed_data/test_data.pt  (optional)
   ```
3. Confirm the data shape matches the expected `(#samples, 1, 48, 48)` for images and `(#samples,)` for labels.

**(B) Baseline**  
```bash
python baseline.py
```
- Loads train/val from `processed_data/`, always predicts “most frequent” class.
- Prints overall accuracy (e.g., ~25%), sets a reference.

**(C) Softmax**  
```bash
python softmax.py
```
- Trains a Softmax (logistic) approach, flattens images to 2304 features.
- Typically yields ~35–40% accuracy.

**(D) Basic NN**  
```bash
python basic_nn.py
```
- Trains a fully connected MLP.  
- Usually better than Softmax, e.g., ~42–45% accuracy.

**(E) Advanced CNN**  
```bash
python advanced_network.py
```
- Trains a convolutional neural network, possibly with weighted cross-entropy, dropout, and early stopping.
- **Saves** final model to `results/advanced_model.pth`.
- Achieves the best performance (often ~55–60% or higher accuracy).

**(F) Live Inference**  
```bash
python live_inference.py
```
- Loads `results/advanced_model.pth`.
- Opens your webcam (index=0).
- Detects face with Haar Cascade, classifies emotion, draws bounding box + label.
- Press **`q`** to quit.

### **5. Desired Output**

- **Console** logs for each training script (loss decreasing, final accuracy, classification reports).
- A **results** folder with:
  - `.txt` files containing final metrics (accuracy, confusion matrix).
  - `advanced_model.pth` (for the advanced CNN).
- **Live Inference** window showing bounding boxes around your face with emotion labels in real-time.

---

## ** General Explanations**

### **1. Project Explanation**

1. **Data**:  
   - The dataset includes images of faces in grayscale, typically 48×48 pixels, each labeled with one of 7 emotions.  
   - The scripts load `.pt` files via PyTorch and use them for training/validation.

2. **Problem Type**:  
   - **Multiclass classification** (7 classes).  
   - The pipeline aims to handle imbalanced data via weighting or other techniques.

3. **Approach**:  
   - Start from a **baseline** to see minimal performance.  
   - Move to **Softmax** for a small improvement.  
   - **Fully connected** network (MLP) adds non-linear capacity.  
   - **Advanced CNN** leverages convolution for better spatial feature extraction.

### **2. Model Summaries**

1. **Baseline**:  
   - Always picks the most frequent class.  
   - Yields ~25% accuracy if “happy” is frequent.

2. **Softmax**:  
   - Linear model with cross-entropy.  
   - Flattens images, sees ~35–40% accuracy.  
   - Underfitting for high-dimensional data.

3. **Basic NN**:  
   - MLP with hidden layers (e.g., 128 or 256 units, ReLU, dropout).  
   - Achieves ~42–45% accuracy. Overfits if not carefully regularized.

4. **Advanced CNN**:  
   - Multiple conv + pool layers, dropout, possibly weighted cross-entropy, early stopping.  
   - Best results (~55–60% or more).  
   - Saves final weights in `advanced_model.pth`.

5. **Live Inference**:  
   - Loads advanced CNN.  
   - Uses Haar Cascade to detect a face.  
   - Crop → 48×48 → CNN → bounding box color + label.  
   - Real-time window with ‘q’ to quit.

### **3. Results Summary**

- **Baseline**: ~25–26%  
- **Softmax**: ~35–38%  
- **Basic NN (MLP)**: ~42–45%  
- **Advanced CNN**: ~55–60%  
- **Live**: Real-time bounding box and predicted label.  
- Overfitting is mitigated by dropout and early stopping in advanced model.  
- Class imbalance (like “disgust”) handled with weighting.

### **4. Conclusion**

- The advanced CNN substantially outperforms simpler methods for facial expressions.  
- Real-time usage is feasible via webcam, though face detection is still required (Haar Cascade or other approaches).  
- Potential improvements:
  - Data augmentation (rotations, flips).  
  - Batch normalization.  
  - Transformers or deeper CNN architectures.

---

## **Appendices**

**A. Code Snippets**  
1. `baseline.py`: Basic model logic, prints metrics.  
2. `softmax.py`: Flatten + linear + cross-entropy.  
3. `basic_nn.py`: MLP with hidden layers, dropout.  
4. `advanced_network.py`: CNN training with early stopping, weight decay.  
5. `live_inference.py`: Real-time classification with bounding boxes using Haar Cascade.

**B. References**  
- PyTorch Documentation: <https://pytorch.org/docs/>  
- OpenCV Haar Cascades: <https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html>  
- CNN Design Patterns: <https://cs231n.github.io/convolutional-networks/>

