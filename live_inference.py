"""
live_inference.py

PURPOSE:
  - Detect faces in webcam frames using Haar Cascade.
  - Classify each detected face as one of 7 emotions using the advanced CNN
    model from advanced_network.py.
  - Draw a bounding box around the face in a color based on the predicted emotion,
    and label it. Press 'q' to quit.

REQUIREMENTS:
  - advanced_network.py => trained model => results/advanced_model.pth
  - haarcascade_frontalface_default.xml (supplied by OpenCV), either:
    - in cv2.data.haarcascades
    - or in the same directory with a full path
  - pip install opencv-python torch numpy

NOTES:
  - This replaces any Keras usage with PyTorch inference:
    face_tensor = ...
    with torch.no_grad():
        logits = self.model(face_tensor)
        pred_idx = torch.argmax(logits).item()
"""

import os
import warnings
import cv2
import torch
import torch.nn as nn
import numpy as np

##############################################################################
#        IGNORE FutureWarning about weights_only=False in torch.load         #
##############################################################################
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

##############################################################################
#               ADVANCED CNN MODEL (From advanced_network.py)                #
##############################################################################
class AdvancedCNN(nn.Module):
    """
    The same architecture used in advanced_network.py, which must match
    the 'advanced_model.pth' weights for correct inference.
    """
    def __init__(self, num_classes=7):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

##############################################################################
#                           MAIN INFERENCE LOOP                              #
##############################################################################
def main():
    """
    1) Load advanced_model.pth from results/.
    2) Open webcam, read frames.
    3) Convert frame to grayscale, detect faces with Haar Cascade, loop each face.
    4) For each face:
       - Crop ROI, resize to 48x48, scale [0..1].
       - Convert to Torch tensor, run forward pass with the advanced CNN.
       - Argmax -> emotion label -> bounding box & text color.
    5) Press 'q' to quit.
    """
    # 1) Build CNN & load weights
    model = AdvancedCNN(num_classes=7)
    model_path = os.path.join("results", "advanced_model.pth")
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found at '{model_path}'. Train advanced_network.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("[Info] Successfully loaded advanced model from results/advanced_model.pth")

    # Emotion dictionary (index -> label). Must match training order
    idx_to_emotion = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Neutral',
        5: 'Sad',
        6: 'Surprise'
    }

    # Color dictionary for bounding boxes (B,G,R)
    color_dict = {
        'Angry':    (0,   0,   255),
        'Disgust':  (0,   255, 0),
        'Fear':     (255, 0,   0),
        'Happy':    (255, 255, 0),
        'Neutral':  (200, 200, 200),
        'Sad':      (128, 0,   0),
        'Surprise': (255, 165, 0)
    }

    # 2) Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open webcam index=0. Exiting.")
        return

    print("[Info] Starting live analysis. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Cannot read frame from camera. Exiting loop.")
            break

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3) Haar cascade for face detection.
        # We assume user has "haarcascade_frontalface_default.xml" under cv2.data.haarcascades
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(face_cascade_path):
            print("[Error] Haar cascade not found. Check your OpenCV installation or path.")
            break

        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop ROI in grayscale
            roi_gray = gray_frame[y:y+h, x:x+w]
            # Resize to 48x48
            roi_gray = cv2.resize(roi_gray, (48, 48))
            # Scale to [0..1]
            roi_gray = roi_gray / 255.0
            # Expand dims to match model input: shape (1,1,48,48)
            roi_tensor = torch.from_numpy(roi_gray).float().unsqueeze(0).unsqueeze(0)

            # Inference with advanced CNN
            with torch.no_grad():
                logits = model(roi_tensor)
                pred_idx = torch.argmax(logits, dim=1).item()
                emotion_label = idx_to_emotion.get(pred_idx, "Unknown")

            # bounding box color
            color = color_dict.get(emotion_label, (255,255,255))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            # Put text above the box
            cv2.putText(frame, emotion_label, (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the result in a window
        cv2.imshow("Live Emotion Analysis (Haar Cascade)", frame)

        # 4) Press 'q' to break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Info] 'q' pressed, exiting live analysis.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Live inference ended. Window closed.")

if __name__ == "__main__":
    main()
