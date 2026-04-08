# FastAPI Detection Backend

## Overview
FastAPI backend for maritime object detection with async support.

## Local Development
```bash
pip install -r requirements.txt
uvicorn fastapi_app:app --reload
```

## Deployment on Render.com
1. Push this folder to GitHub repository
2. Go to Render.com dashboard
3. Click "New +" -> "Web Service"
4. Connect your GitHub repository
5. Runtime: Python 3
6. Build Command: `pip install -r requirements.txt`
7. Start Command: `gunicorn -k uvicorn.workers.UvicornWorker fastapi_app:app`

## API Endpoints
- `POST /api/detect?image_id=string`
  - Response: `{"image_id": "string", "detections": [...]}`

## Features
- Async/await support
- Pydantic data validation
- Automatic API docs at `/docs`
- CORS enabled for all origins
