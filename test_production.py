import requests
from PIL import Image, ImageDraw
import io

# Create test image
img = Image.new('RGB', (640, 480), color='white')
draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=3)
draw.ellipse([300, 150, 400, 250], fill='orange', outline='black', width=3)

# Save image
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
image_bytes = buffer.getvalue()

print("Testing Production API on Render...")
print("URL: https://fastapi-detect-project.onrender.com")

# Test production API
files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
data = {'confidence_threshold': 0.3}

try:
    response = requests.post('https://fastapi-detect-project.onrender.com/api/detect-upload', files=files, data=data)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Objects: {result['total_detections']}")
        print(f"Processing time: {result['processing_time']:.4f}s")
        print(f"Model: {result['model_info']['name']}")
        print(f"Mode: {result['model_info']['type']}")
        
        for i, det in enumerate(result['detections']):
            class_name = det.get('class_name', det.get('class', 'unknown'))
            print(f"  {i+1}. {class_name} (conf: {det['confidence']:.3f})")
            print(f"     Color: {det['color']}")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"Exception: {e}")

print("\nProduction API test completed!")
