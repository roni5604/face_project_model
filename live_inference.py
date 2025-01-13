"""
live_inference.py

PURPOSE:
  - Perform real-time facial expression classification from your webcam feed using the
    advanced CNN that has the FIRST conv kernel=5, second conv kernel=3, etc.
  - This matches advanced_network.py exactly to avoid shape mismatch errors.
  - Detects faces with Haar cascade, resizes them to 48x48, passes to CNN,
    draws bounding boxes with emotion labels.

REQUIREMENTS:
  - PyTorch, OpenCV, and the same advanced_model.pth from advanced_network.py
  - Haar cascade file for face detection (in cv2.data.haarcascades or specify path).
"""

import os
import warnings
import cv2
import torch
import torch.nn as nn
import numpy as np

##############################################################################
# 0) SUPPRESS FUTUREWARNING FROM TORCH.LOAD
##############################################################################
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

##############################################################################
# 1) ADVANCED CNN MATCHING advanced_network.py
##############################################################################
class AdvancedEmotionCNN(nn.Module):
    """
    Must match advanced_network.py.
    The first conv has kernel=5, padding=2 => out=64
    The second conv has kernel=3, ...
    Then two more conv blocks, then flatten => 4608 => (256->512->7).
    Dropout(0.25), BN, ReLU throughout.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 1st conv block => kernel=5
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # 2nd conv block => kernel=3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # 3rd conv block => kernel=3
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # 4th conv block => kernel=3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            # flatten
            nn.Flatten(),

            # FC1
            nn.Linear(in_features=512*3*3, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            # FC2
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            # final => 7 classes
            nn.Linear(512, 7)
        )

    def forward(self, x):
        return self.network(x)

##############################################################################
# 2) MAIN: LIVE INFERENCE
##############################################################################
def main():
    """
    1) Build the same advanced CNN (first conv kernel=5, etc.).
    2) Load 'results/advanced_model.pth'.
    3) Open webcam, detect faces with Haar cascade, classify each face.
    4) Draw bounding boxes + label in color. Press 'q' to quit.
    """
    # A) Create same model
    model = AdvancedEmotionCNN()

    # B) Load weights
    model_file = "results/advanced_model.pth"
    if not os.path.exists(model_file):
        print("[Error] No model found at", model_file)
        return

    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()
    print("[Info] Loaded advanced CNN from", model_file)

    # C) Map indices -> emotions
    idx_to_emo = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprise"
    }

    color_map = {
        "Angry":    (0,   0,   255),
        "Disgust":  (0,   255, 0),
        "Fear":     (255, 0,   0),
        "Happy":    (255, 255, 0),
        "Neutral":  (200, 200, 200),
        "Sad":      (128, 0,   0),
        "Surprise": (255, 165, 0)
    }

    # D) Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam index=0.")
        return
    print("[Info] Starting webcam. Press 'q' to exit.\n")

    # E) Haar Cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print("[Error] Haar cascade not found. Check OpenCV.")
        return
    face_cascade = cv2.CascadeClassifier(cascade_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Could not read frame. Exiting loop.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30,30)
        )

        for (x,y,w,h) in faces:
            # crop
            face_roi = gray_frame[y:y+h, x:x+w]
            # resize => 48x48
            face_roi = cv2.resize(face_roi, (48,48), interpolation=cv2.INTER_AREA)
            # to float tensor => (1,1,48,48)
            face_tensor = torch.from_numpy(face_roi).float().unsqueeze(0).unsqueeze(0)
            face_tensor = face_tensor / 255.0  # scale [0..1]

            # inference
            with torch.no_grad():
                logits = model(face_tensor)  # shape => (1,7)
                pred_idx = torch.argmax(logits, dim=1).item()

            emotion_name = idx_to_emo.get(pred_idx, "Unknown")
            color = color_map.get(emotion_name, (255,255,255))

            # draw bounding box + text
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, emotion_name, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # show
        cv2.imshow("Facial Expression - Advanced CNN Live", frame)

        # Press 'q' => quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Info] 'q' pressed, closing.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Webcam window closed. End of live inference.")

##############################################################################
# 3) SCRIPT ENTRY
##############################################################################
if __name__ == "__main__":
    main()
