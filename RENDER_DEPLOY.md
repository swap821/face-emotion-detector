# Deploy to Render (Backend)

1. Go to https://dashboard.render.com/
2. Click "New +" → "Web Service"
3. Connect GitHub repo: `swap821/face-emotion-detector`
4. Configure:
   - **Name**: `face-emotion-detector-api`
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. Add Environment Variable:
   - `ALLOWED_ORIGINS` = `https://face-emotion-detector.vercel.app,http://localhost:5173`
6. Click "Create Web Service"

Your backend URL will be: `https://face-emotion-detector-api.onrender.com`

**Note**: This project requires the FER2013 dataset for training. For deployment, either:
- Train locally and commit the `.h5` model file, OR
- Download dataset during build (slow), OR
- Use a pre-trained model
