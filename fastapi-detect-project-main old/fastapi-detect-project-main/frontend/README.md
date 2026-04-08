# FastAPI Detection Frontend

## Overview
Static HTML frontend for FastAPI maritime detection application.

## Local Development
Open `detect_web.html` directly in browser or use live server.

## Deployment on Vercel
1. Push this folder to GitHub repository
2. Go to Vercel.com dashboard
3. Click "New Project"
4. Import your GitHub repository
5. Vercel will auto-detect static site
6. Deploy

## Configuration
The frontend is configured to call the production backend at:
`https://fastapi-detect-backend.onrender.com`

For local testing, change the API_BASE_URL in detect_web.html to:
`http://127.0.0.1:8000`

## Files
- `detect_web.html` - Main application interface
- `vercel.json` - Vercel deployment configuration
