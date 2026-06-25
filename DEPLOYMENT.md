# Deployment Guide - Face Emotion Detector

## Quick Deploy (Render + Vercel)

### Backend (Render)

1. Go to [render.com](https://render.com) → "New Web Service"
2. Connect your GitHub repo: `swap821/face-emotion-detector`
3. Configure:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
4. Click "Create Web Service"
5. Copy the service URL (e.g., `https://face-emotion-detector.onrender.com`)

### Frontend (Vercel)

1. Go to [vercel.com](https://vercel.com) → "Add New Project"
2. Import your GitHub repo: `swap821/face-emotion-detector`
3. Set **Framework Preset** to "Vite"
4. Set **Root Directory** to `frontend`
5. Add Environment Variable:
   - `VITE_API_URL` = your Render backend URL
6. Click "Deploy"

### One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/swap821/face-emotion-detector)

---

## Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
python train_cnn.py         # Trains CNN model (requires FER2013 dataset)
python app.py               # Starts Flask server on localhost:5000

# Frontend
cd frontend
npm install
npm run dev                 # Starts dev server on localhost:3000
```

## Dataset Download

The FER2013 dataset is required for training:
- Download from [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Extract to `backend/data/fer2013/` with `train/` and `test/` subdirectories
- Or use the pre-trained model if available

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FLASK_ENV` | No | Set to `production` for deployment |
| `ALLOWED_ORIGINS` | Yes | Comma-separated CORS origins |
| `GEMINI_API_KEY` | No | Optional, for enhanced emotion insights |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/emotion` | POST | Detect emotion from uploaded image |
| `/api/emotion/live` | WebSocket | Real-time emotion from webcam stream |
| `/api/health` | GET | Health check |

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'cv2'`
**Fix**: Run `pip install opencv-python opencv-python-headless`

**Issue**: FER2013 dataset not found
**Fix**: Download from Kaggle link above, or use a pre-trained model

**Issue**: Webcam not working in browser
**Fix**: Ensure you're using HTTPS (required for getUserMedia). Use localhost for development.
