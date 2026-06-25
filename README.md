# Face Emotion Detector

**Live Demo:** [https://63dcmrawskzy4.kimi.page](https://63dcmrawskzy4.kimi.page)

Real-time facial emotion detection using OpenCV and CNN. Identifies 7 emotions from webcam video with WebSocket streaming.

## Features
- **Real-time processing**: 5 FPS with WebSocket streaming
- **7 emotions**: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- **CNN from scratch**: 4 convolutional blocks + 2 dense layers
- **WebSocket**: Low-latency streaming

## Architecture
```
Webcam → OpenCV Face Detection → CNN (48x48 grayscale) → 7 Emotion Classes
```

## Tech Stack
- Python, TensorFlow, Keras, OpenCV
- Flask, Flask-SocketIO
- React, WebSocket

## Quick Start
```bash
cd backend
pip install -r requirements.txt
python test_camera.py   # Test webcam first
python train_cnn.py     # Train model (requires FER2013 dataset)
python app.py           # API on localhost:5000
cd ../frontend
npm install && npm run dev
```

## Deploy
- Backend: [Deploy to Render](https://render.com/deploy?repo=https://github.com/swap821/face-emotion-detector)
- See `DEPLOYMENT.md` for full instructions

## Author
**Swapnil Kumar** — [Portfolio](https://swapnil-kumar-portfolio016.vercel.app) | [GitHub](https://github.com/swap821)
