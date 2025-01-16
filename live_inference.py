"""
live_inference.py

PURPOSE:
  - Provide a real-time facial expression classification interface using a webcam.
  - Loads the advanced CNN model architecture from 'advanced_network.py' and the
    trained weights from 'results/advanced_model.pth'.
  - Uses OpenCV to capture frames, Haar cascade to detect faces, resizes each face
    to 48x48, applies the same normalization as in dataset_preparation, then
    passes it through the advanced CNN to predict an emotion.

GENERAL EXPLANATIONS:
  1) This script is how we "go live" with our advanced CNN after it has been trained.
  2) The advanced CNN is "advanced" because it has multiple conv blocks, batchnorm,
     dropout, etc. specialized for images (unlike baseline, single-layer softmax,
     or basic MLP).
  3) This script:
      - Captures frames from webcam (cv2.VideoCapture(0)).
      - Detects faces with HaarCascade (frontalface_default).
      - For each face: Crop => resize => convert to tensor => feed model => argmax => label.
      - Draws bounding boxes around the face + writes the predicted emotion label in color.

HOW TO RUN:
  python live_inference.py

REQUIREMENTS:
  - A working webcam.
  - OpenCV (pip install opencv-python).
  - The advanced model's architecture must match 'advanced_network.py'.
  - The file 'results/advanced_model.pth' from a trained advanced CNN must exist.
  - HaarCascade file: Usually found in cv2.data.haarcascades, e.g. "haarcascade_frontalface_default.xml".

NOTES:
  - Press 'q' to quit the window.
  - If not able to open the webcam, check your camera index or drivers.
"""

import os
import warnings
import cv2
import torch
import torch.nn as nn
import numpy as np

##############################################################################
# 0) SUPPRESS FUTUREWARNING FROM TORCH.LOAD
# Explanation: we ignore 'weights_only=False' warnings as we want to load full model states.
##############################################################################
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)


##############################################################################
# 1) ADVANCED CNN MODEL (MUST MATCH advanced_network.py EXACTLY)
##############################################################################
class AdvancedEmotionCNN(nn.Module):
    """
    The same advanced CNN as in advanced_network.py:
      - 4 conv blocks (Conv2d + BN + ReLU + MaxPool + Dropout)
      - flatten => 2 FC layers + BN + Dropout => final 7-class output

    The first conv block uses kernel=5, all others kernel=3,
    with careful in_channels/out_channels to avoid mismatch.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 1st conv block => kernel=5, out_channels=64
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            # 2nd conv block => kernel=3, out_channels=128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.25),

            # 3rd conv block => kernel=3, out_channels=512
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.25),

            # 4th conv block => kernel=3, out_channels=512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.25),

            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(512, 7)
        )

    def forward(self, x):
        return self.network(x)


##############################################################################
# 2) MAIN LIVE INFERENCE LOOP
##############################################################################
def main():
    """
    Steps:
      1) Build the same advanced CNN as 'advanced_network.py'.
      2) Load 'results/advanced_model.pth' (trained weights).
      3) Initialize webcam capture (index=0 by default).
      4) Use Haar cascade to detect faces in each frame.
      5) For each face:
           - crop & resize => 48x48
           - convert to float => normalize [0..1] or similar
           - forward pass => argmax => emotion label
      6) Draw bounding box & label on the frame, show in a window.
      7) Press 'q' to quit.

    This is the final step where the advanced CNN is tested in real time after training.
    """
    # A) Create the advanced CNN (exact same structure as in advanced_network.py).
    model = AdvancedEmotionCNN()

    # B) Load the final trained weights from 'results/advanced_model.pth'
    model_path = "results/advanced_model.pth"
    if not os.path.exists(model_path):
        print("[Error] advanced_model.pth not found. Make sure you have trained 'advanced_network.py' first.")
        return

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("[Info] Successfully loaded advanced CNN from", model_path)

    # Map class indices to emotion names
    idx_to_emotion = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprise"
    }

    # We'll define a color dictionary for bounding boxes
    color_map = {
        "Angry":    (0,   0,   255),
        "Disgust":  (0,   255, 0),
        "Fear":     (255, 0,   0),
        "Happy":    (255, 255, 0),
        "Neutral":  (192, 192, 192),
        "Sad":      (128, 0,   0),
        "Surprise": (255, 165, 0)
    }

    # C) Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open webcam index=0. Exiting.")
        return

    print("[Info] Webcam opened. Press 'q' to quit the window.\n")

    # D) Haar Cascade for face detection
    # Usually present in cv2.data.haarcascades
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print("[Error] Haar cascade not found. Check your OpenCV installation or path.")
        return
    face_cascade = cv2.CascadeClassifier(cascade_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Could not read frame from camera. Exiting loop.")
            break

        # Convert to grayscale for cascade detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces (scaleFactor and minNeighbors can be tuned)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # Crop region of interest from the grayscale frame
            face_roi = gray_frame[y:y+h, x:x+w]
            # Resize => 48x48 to match the training dimension
            face_roi = cv2.resize(face_roi, (48,48), interpolation=cv2.INTER_AREA)

            # Convert to torch tensor of shape (1,1,48,48)
            face_tensor = torch.from_numpy(face_roi).float().unsqueeze(0).unsqueeze(0)
            # Scale to [0..1], or we can do face_tensor/255 then normalize like in dataset_preparation
            face_tensor = face_tensor / 255.0

            with torch.no_grad():
                logits = model(face_tensor)
                pred_idx = torch.argmax(logits, dim=1).item()

            # Determine emotion name and color
            emotion_name = idx_to_emotion.get(pred_idx, "Unknown")
            color = color_map.get(emotion_name, (255,255,255))

            # Draw bounding box
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            # Put text above the rectangle
            cv2.putText(frame, emotion_name, (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Show the result
        cv2.imshow("Live Facial Expression Detection (Advanced CNN)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Info] 'q' pressed, closing live inference.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Webcam window closed. End of live analysis.")


if __name__ == "__main__":
    main()
