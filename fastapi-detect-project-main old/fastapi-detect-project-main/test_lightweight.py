import requests
import base64
from PIL import Image, ImageDraw
import io

# Tạo ảnh test với hình dạng đơn giản
img = Image.new('RGB', (640, 480), color='white')
draw = ImageDraw.Draw(img)

# Vẽ các hình dạng khác nhau để test detection
draw.rectangle([100, 100, 180, 180], fill='red', outline='black', width=3)      # Square ~ apple
draw.rectangle([250, 150, 350, 250], fill='yellow', outline='black', width=3)  # Rectangle ~ banana  
draw.ellipse([400, 200, 480, 320], fill='orange', outline='black', width=3)  # Circle ~ orange
draw.rectangle([150, 300, 190, 380], fill='purple', outline='black', width=3) # Small ~ grape

# Lưu ảnh
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
image_bytes = buffer.getvalue()

print("🚀 Testing Lightweight Detection...")
print("📤 Sending test image with shapes...")

# Test YOLOv8 mode
files = {'image': ('test_shapes.jpg', image_bytes, 'image/jpeg')}
data = {'confidence_threshold': 0.3, 'use_lightweight': False}

try:
    response = requests.post('http://127.0.0.1:8000/api/detect-upload', files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n🔥 YOLOv8 Mode:")
        print(f"✅ Objects found: {result['total_detections']}")
        print(f"⏱️  Processing time: {result['processing_time']:.4f}s")
        print(f"🎯 Classes: {result['detected_classes']}")
        
        for i, det in enumerate(result['detections']):
            print(f"  {i+1}. {det['class_name']} (confidence: {det['confidence']:.3f})")
    else:
        print(f"❌ YOLOv8 Error: {response.status_code}")
        
except Exception as e:
    print(f"❌ YOLOv8 Exception: {e}")

# Test Lightweight mode
data_lightweight = {'confidence_threshold': 0.3, 'use_lightweight': True}

try:
    response = requests.post('http://127.0.0.1:8000/api/detect-upload', files=files, data=data_lightweight)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n⚡ Lightweight Mode:")
        print(f"✅ Objects found: {result['total_detections']}")
        print(f"⏱️  Processing time: {result['processing_time']:.4f}s")
        print(f"🎯 Classes: {result['detected_classes']}")
        print(f"🔧 Model: {result['model_info']['name']}")
        
        for i, det in enumerate(result['detections']):
            class_name = det.get('class_name', det.get('class', 'unknown'))
            print(f"  {i+1}. {class_name} (confidence: {det['confidence']:.3f})")
            print(f"     Bbox: ({det['bbox']['x_min']}, {det['bbox']['y_min']}) → ({det['bbox']['x_max']}, {det['bbox']['y_max']})")
            print(f"     Color: {det['color']}")
            
    else:
        print(f"❌ Lightweight Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Lightweight Exception: {e}")

print("\n🎯 Comparison completed!")
print("💡 Lightweight mode should be faster and suitable for Vercel/Render")
