# FastAPI Detection Project

## Project Structure
```
fastapi-detect-project-new/
backend/          # FastAPI (Deploy to Render)
frontend/         # Static HTML (Deploy to Vercel)
```

## Quick Start

### Backend (Local)
```bash
cd backend
pip install -r requirements.txt
uvicorn fastapi_app:app --reload
```

### Frontend (Local)
Open `frontend/detect_web.html` in browser.

## Production Deployment

### Backend on Render.com
1. Push entire project to GitHub
2. On Render: Connect repository
3. Root Directory: `backend`
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `gunicorn -k uvicorn.workers.UvicornWorker fastapi_app:app`

### Frontend on Vercel.com
1. Push entire project to GitHub
2. On Vercel: Connect repository
3. Root Directory: `frontend`
4. Build Command: (empty for static)
5. Output Directory: (empty for static)

## URLs After Deployment
- Backend: `https://your-app.onrender.com`
- Frontend: `https://your-app.vercel.app`

## API Configuration
Frontend is configured to call production backend. For local testing, change `API_BASE_URL` in `frontend/detect_web.html`.

## API Endpoint
- `POST /api/detect?image_id=string`
  - Response: `{"image_id": "string", "detections": [...]}`

## Features
- Async/await support
- Pydantic data validation
- Automatic API docs at `/docs`
