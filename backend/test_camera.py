"""
test_camera.py — Standalone Webcam Test

Tests that OpenCV can access the webcam and detect faces.
Run this BEFORE training the model to verify your setup.

Usage: python test_camera.py
Press 'q' to quit.
"""

import cv2


def test_webcam():
    """Open webcam and draw rectangles around detected faces."""
    # Load Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        print("Make sure you have a webcam connected and permissions granted")
        return
    
    print("Webcam test started! Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Webcam Test - Press q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test complete!")


if __name__ == "__main__":
    test_webcam()