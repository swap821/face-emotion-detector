"""
app.py — Flask API + WebSocket for Real-Time Emotion Detection

Endpoints:
- POST /predict_frame: Analyze base64 image
- WS /stream: Real-time WebSocket streaming
- GET /emotions: List supported emotions
- GET /models/info: Model info
- GET /health: Health check
"""

import os
import pickle
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO

from face_detector import detect_faces, preprocess_face
from emotion_labels import EMOTIONS, get_emotion_info
from utils import setup_logging

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

logger = setup_logging()

MODEL = None
CLASS_INDICES = None

def load_model_on_startup():
    global MODEL, CLASS_INDICES
    try:
        MODEL = load_model('models/emotion_model.h5')
        import json
        with open('models/class_indices.json', 'r') as f:
            CLASS_INDICES = json.load(f)
        logger.info("Emotion model loaded")
    except:
        logger.warning("Model not found. Run train_cnn.py first.")

load_model_on_startup()


@app.route('/')
def home():
    return jsonify({
        'message': 'Face Emotion Detector API',
        'models_loaded': MODEL is not None,
        'endpoints': {
            'POST /predict_frame': 'Analyze base64 image',
            'GET /emotions': 'List emotions',
            'GET /health': 'Health check'
        }
    })


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.get_json()
    img_data = base64.b64decode(data['image'].split(',')[1])
    img = Image.open(BytesIO(img_data))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    faces = detect_faces(frame)
    results = []
    
    for (x, y, w, h, face_roi) in faces:
        processed = preprocess_face(face_roi)
        prediction = MODEL.predict(processed, verbose=0)[0]
        emotion_idx = np.argmax(prediction)
        confidence = float(prediction[emotion_idx])
        
        info = get_emotion_info(emotion_idx)
        results.append({
            'emotion': info['name'],
            'emoji': info['emoji'],
            'confidence': round(confidence, 4),
            'bounding_box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })
    
    return jsonify({'faces_detected': len(results), 'results': results})


@app.route('/emotions', methods=['GET'])
def get_emotions():
    return jsonify({'emotions': [
        {'index': k, **v} for k, v in EMOTIONS.items()
    ]})


@app.route('/models/info', methods=['GET'])
def model_info():
    return jsonify({
        'architecture': 'CNN with 4 Conv blocks + 2 Dense layers',
        'input_shape': '(48, 48, 1)',
        'num_classes': 7,
        'classes': list(EMOTIONS.values())
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': MODEL is not None})


@socketio.on('frame')
def handle_frame(data):
    """WebSocket handler for real-time frames."""
    if MODEL is None:
        emit('error', {'message': 'Model not loaded'})
        return
    
    try:
        img_data = base64.b64decode(data.split(',')[1] if ',' in data else data)
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        faces = detect_faces(frame)
        results = []
        
        for (x, y, w, h, face_roi) in faces:
            processed = preprocess_face(face_roi)
            prediction = MODEL.predict(processed, verbose=0)[0]
            emotion_idx = np.argmax(prediction)
            confidence = float(prediction[emotion_idx])
            info = get_emotion_info(emotion_idx)
            results.append({
                'emotion': info['name'],
                'emoji': info['emoji'],
                'confidence': round(confidence, 4),
                'box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })
        
        emit('prediction', {'faces': len(results), 'results': results})
    except Exception as e:
        emit('error', {'message': str(e)})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)