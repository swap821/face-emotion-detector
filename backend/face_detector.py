"""
face_detector.py — OpenCV Face Detection

Uses Haar Cascade classifiers to detect faces in video frames.
The Viola-Jones algorithm works by scanning the image with a sliding window
and checking for facial features (eyes, nose, mouth) at multiple scales.
"""

import cv2
import numpy as np

# Load the Haar Cascade face detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def detect_faces(frame):
    """
    Detect faces in a video frame.
    
    Args:
        frame: BGR image from webcam
        
    Returns:
        list: [(x, y, w, h, face_roi), ...] for each detected face
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,   # How much image size is reduced at each scale
        minNeighbors=5,     # How many neighbors each candidate rectangle should have
        minSize=(48, 48),   # Minimum face size
    )
    
    results = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        results.append((x, y, w, h, face_roi))
    
    return results


def draw_box(frame, x, y, w, h, emotion, confidence):
    """Draw bounding box and label on frame."""
    from emotion_labels import get_emotion_color
    color = tuple(int(get_emotion_color(emotion)[i:i+2], 16) for i in (1, 3, 5))
    color = (color[2], color[1], color[0])  # RGB to BGR
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    label = f"{EMOTIONS[emotion]['emoji']} {EMOTIONS[emotion]['name']} ({confidence:.0f}%)"
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def preprocess_face(face_roi):
    """
    Preprocess face ROI for CNN input.
    
    Steps:
    1. Resize to 48x48 (FER-2013 image size)
    2. Normalize pixel values to [0, 1]
    3. Reshape for model input (1, 48, 48, 1)
    
    Args:
        face_roi: Grayscale face region
        
    Returns:
        numpy.ndarray: Preprocessed face (1, 48, 48, 1)
    """
    resized = cv2.resize(face_roi, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped