import requests
from PIL import Image, ImageDraw
import io

# TCreate simple test image
img = Image.new('RGB', (640, 480), color='white')
draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=3)

# Save image
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
image_bytes = buffer.getvalue()

print("Testing lightweight mode only...")

# Test only lightweight mode
files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
data = {'confidence_threshold': 0.3, 'use_lightweight': True}

try:
    response = requests.post('http://127.0.0.1:8000/api/detect-upload', files=files, data=data)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Objects: {result['total_detections']}")
        print(f"Time: {result['processing_time']:.4f}s")
        model_info = result.get('model_info', {})
        print(f"Model: {model_info.get('name', 'Unknown')}")
        
        for i, det in enumerate(result['detections']):
            class_name = det.get('class_name', det.get('class', 'unknown'))
            print(f"  {i+1}. {class_name} (conf: {det['confidence']:.3f})")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
