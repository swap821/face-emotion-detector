# 😊 Face Emotion Detector

**Live Demo:** [https://face-emotion-detector.vercel.app](https://face-emotion-detector.vercel.app) *(deploy after setup)*

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Socket.io](https://img.shields.io/badge/Socket.io-black?style=for-the-badge&logo=socket.io&badgeColor=010101)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

A real-time facial emotion detection application that identifies **7 emotions** — Happy, Sad, Angry, Surprise, Fear, Disgust, and Neutral — directly from webcam video using a **Convolutional Neural Network (CNN)**. Features WebSocket streaming for low-latency live analysis with OpenCV face detection preprocessing.

This project demonstrates computer vision, deep learning for image classification, real-time video processing, and WebSocket-based bidirectional communication.

---

# ✨ Key Features

## 🎥 Real-Time Emotion Detection
- Live webcam feed analysis at ~5 FPS
- WebSocket streaming for minimal latency
- Face detection with OpenCV Haar Cascade
- Bounding box + emotion label overlay on detected faces

## 🧠 CNN Architecture (from scratch)
- 4 convolutional blocks with BatchNorm + ReLU + MaxPool
- 2 dense layers with Dropout regularization
- 7-class softmax output (FER2013 emotion classes)
- Input: 48×48 grayscale face images

## 😊 7 Emotion Classes
| Emotion | Emoji | Description |
|---------|-------|-------------|
| Happy | 😊 | Joy, pleasure |
| Sad | 😢 | Sorrow, grief |
| Angry | 😠 | Rage, annoyance |
| Surprise | 😲 | Astonishment |
| Fear | 😨 | Worry, terror |
| Disgust | 🤢 | Aversion, loathing |
| Neutral | 😐 | No strong emotion |

## 🖥️ Interactive React Dashboard
- Live webcam preview with emotion overlay
- Emotion distribution bar chart
- Confidence score for each prediction
- Toggle between image upload and webcam modes

---

# 🛠️ Tech Stack

## Backend (`/backend`)
- Python 3.11
- Flask + Flask-SocketIO (WebSocket server)
- TensorFlow / Keras (CNN model)
- OpenCV (face detection & image preprocessing)
- NumPy (array operations)

## Frontend (`/frontend`)
- React 18
- Vite
- Tailwind CSS
- Socket.IO Client (real-time WebSocket)
- Recharts (emotion distribution charts)

---

# 📂 Project Structure

```plaintext
face-emotion-detector/
│
├── backend/                     # Flask + WebSocket Server
│   ├── models/                  # Trained CNN model (.h5)
│   ├── data/                    # FER2013 dataset (download separately)
│   ├── train_cnn.py             # CNN training pipeline
│   ├── face_detector.py         # OpenCV face detection & preprocessing
│   ├── test_camera.py           # Standalone webcam test
│   ├── app.py                   # Flask + SocketIO server
│   ├── requirements.txt         # Python dependencies
│   └── Procfile                 # Render deployment config
│
├── frontend/                    # React Application
│   ├── src/
│   │   ├── components/          # UI Components
│   │   │   ├── WebcamView.jsx       # Live webcam + Socket.IO
│   │   │   ├── EmotionChart.jsx     # Emotion distribution
│   │   │   └── ImageUpload.jsx      # Static image analysis
│   │   ├── App.jsx              # Main Application
│   │   └── main.jsx             # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── render.yaml                  # Render blueprint
├── vercel.json                  # Vercel deployment config
└── README.md
```

---

# 🚀 Getting Started

## 📌 Prerequisites
- Python 3.11+
- Node.js 18+
- Webcam (for live detection)
- FER2013 dataset (for training, ~60MB)

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/swap821/face-emotion-detector.git
cd face-emotion-detector
```

---

# 🔧 Backend Setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📥 Download FER2013 Dataset

Download from [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013) and extract to:
```
backend/data/fer2013/
├── train/     # Training images organized by emotion
└── test/      # Test images organized by emotion
```

---

## 🏋️ Train the CNN Model

```bash
# Test webcam first
python test_camera.py

# Train the CNN (this will take 10-20 minutes on CPU)
python train_cnn.py
```

**Output:** Saves `emotion_model.h5` in `backend/models/`

**CNN Architecture:**
```
Input (48×48×1)
→ Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
→ Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
→ Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
→ Conv2D(512) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
→ Flatten → Dense(256) → BatchNorm → ReLU → Dropout(0.5)
→ Dense(7, softmax) → Output
```

---

## ▶️ Start the Flask + WebSocket Server

```bash
python app.py
```

API runs at `http://localhost:5000`
WebSocket at `ws://localhost:5000/stream`

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/emotion` | POST | Detect emotion from uploaded image |
| `/api/emotion/live` | WebSocket | Real-time emotion from webcam frames |
| `/api/health` | GET | Health check |

---

# 🎨 Frontend Setup

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

---

# 🌍 Deployment

## Backend — Render

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub repo: `swap821/face-emotion-detector`
4. Configure:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. Add Environment Variable:
   - `ALLOWED_ORIGINS` = `https://face-emotion-detector.vercel.app,http://localhost:5173`
6. Click **Create Web Service**

> **Note:** For deployment, train the model locally first and commit the `.h5` file, or include the FER2013 dataset in your repo for training during build.

## Frontend — Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click **Add New Project** → Import `face-emotion-detector`
3. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
4. Add Environment Variable:
   - `VITE_API_URL` = `https://face-emotion-detector-api.onrender.com`
5. Click **Deploy**

---

# 🧠 Computer Vision & DL Concepts Demonstrated

This project demonstrates understanding of:

- **Convolutional Neural Networks**: Feature extraction with Conv2D layers
- **Batch Normalization**: Stabilizing training and accelerating convergence
- **Dropout Regularization**: Preventing overfitting
- **Image Preprocessing**: Grayscale conversion, resizing, normalization
- **Face Detection**: OpenCV Haar Cascade for real-time face localization
- **WebSocket Communication**: Bidirectional streaming for live video
- **Real-Time Inference**: Optimizing model for low-latency predictions
- **Multi-Class Classification**: Softmax for 7 emotion categories

---

# 🚀 Future Improvements

- [ ] Fine-tune on custom emotion dataset
- [ ] Add age and gender estimation
- [ ] Implement attention mechanism for better accuracy
- [ ] Mobile app with React Native
- [ ] Edge deployment with TensorFlow Lite
- [ ] Multi-face detection and tracking
- [ ] Emotion intensity scoring (0-100%)
- [ ] Record and analyze emotion over time

---

# 👨‍💻 Author

## Swapnil Kumar

- GitHub: https://github.com/swap821
- LinkedIn: https://www.linkedin.com/in/swapnil-kumar-73a68a308
- Portfolio: https://swapnil-kumar-portfolio016.vercel.app

---

# ⭐ Project Goal

This project was built to demonstrate:
- Convolutional Neural Network design from scratch
- Real-time computer vision with OpenCV
- WebSocket streaming for live video processing
- Full-stack CV application architecture
- Image classification for multi-class problems
- Production deployment of deep learning models

---

# 📜 License

This project is open-source and available for educational and learning purposes.
