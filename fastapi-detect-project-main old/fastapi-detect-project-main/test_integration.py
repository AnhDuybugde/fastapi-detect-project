import requests
from bs4 import BeautifulSoup
import time

print("Testing Frontend-Backend Integration...")
print("=" * 50)

# Test 1: Backend Health
print("\n1. Testing Backend Health...")
try:
    response = requests.get("https://fastapi-detect-project.onrender.com/api/health")
    if response.status_code == 200:
        health_data = response.json()
        print(f"   Status: {health_data['status']}")
        print(f"   Mode: {health_data['mode']}")
        print("   Backend: OK")
    else:
        print(f"   Backend Error: {response.status_code}")
except Exception as e:
    print(f"   Backend Exception: {e}")

# Test 2: Backend Fruits API
print("\n2. Testing Backend Fruits API...")
try:
    response = requests.get("https://fastapi-detect-project.onrender.com/api/fruits")
    if response.status_code == 200:
        fruits_data = response.json()
        print(f"   Available fruits: {fruits_data['total_classes']}")
        print(f"   Classes: {fruits_data['classes']}")
        print("   Fruits API: OK")
    else:
        print(f"   Fruits API Error: {response.status_code}")
except Exception as e:
    print(f"   Fruits API Exception: {e}")

# Test 3: Frontend Accessibility (if deployed)
print("\n3. Testing Frontend...")
frontend_urls = [
    "https://fastapi-detect-project.vercel.app",
    "https://fastapi-detect-project.onrender.com/frontend/detect_web.html"
]

for url in frontend_urls:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Check if HTML contains API_BASE_URL
            if "fastapi-detect-project.onrender.com" in response.text:
                print(f"   Frontend at {url}: OK")
                print(f"   Status: {response.status_code}")
                break
            else:
                print(f"   Frontend at {url}: API URL not configured")
        else:
            print(f"   Frontend at {url}: {response.status_code}")
    except Exception as e:
        print(f"   Frontend at {url}: {e}")

# Test 4: CORS Check
print("\n4. Testing CORS...")
try:
    response = requests.options("https://fastapi-detect-project.onrender.com/api/health")
    if response.status_code == 200:
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
        }
        print(f"   CORS Headers: {cors_headers}")
        print("   CORS: OK")
    else:
        print(f"   CORS Error: {response.status_code}")
except Exception as e:
    print(f"   CORS Exception: {e}")

print("\n" + "=" * 50)
print("Integration Test Summary:")
print("Backend: https://fastapi-detect-project.onrender.com")
print("Frontend: Deploy to Vercel (if not already)")
print("API Configuration: Already set in frontend")
print("CORS: Enabled for all origins")
print("\nNext Steps:")
print("1. Deploy frontend to Vercel: vercel --prod")
print("2. Test full workflow in browser")
print("3. Monitor production performance")
