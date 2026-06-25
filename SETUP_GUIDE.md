# Setup Guide — Face Emotion Detector

## Prerequisites
- Python 3.9+, Node.js 18+, Webcam

## Step 1: Download FER-2013 Dataset
https://www.kaggle.com/datasets/msambare/fer2013

## Step 2: Test Camera
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python test_camera.py  # Should show green boxes around faces
```

## Step 3: Train CNN
```bash
python train_cnn.py  # 30-60 min on CPU
```

## Step 4: Run
```bash
python app.py  # Backend
cd ../frontend && npm install && npm run dev  # Frontend
```