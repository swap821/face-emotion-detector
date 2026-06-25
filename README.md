# Face Emotion Detector — Real-Time Facial Expression Recognition

## Overview
Real-time facial emotion detection using OpenCV and CNN. Identifies 7 emotions from webcam video.

## Features
- **Real-time processing**: 5 FPS with WebSocket streaming
- **7 emotions**: 😊😢😠😲😨🤢😐
- **CNN from scratch**: 4 convolutional blocks
- **WebSocket**: Low-latency streaming

## Tech Stack
- OpenCV, Haar Cascade, TensorFlow/Keras CNN
- Flask, Flask-SocketIO, React

## Quick Start
```bash
cd backend
pip install -r requirements.txt
python test_camera.py  # Test webcam first
python train_cnn.py    # Train model (30-60 min)
python app.py          # Start API
cd ../frontend && npm install && npm run dev
```

## Architecture
```
Webcam → OpenCV Face Detection → CNN (48x48) → 7 Emotion Classes
```

## Author
**Swapnil Kumar** — https://github.com/swap821