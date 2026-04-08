import requests
import base64
from PIL import Image, ImageDraw
import io

# Tạo ảnh test đơn giản với hình vuông
img = Image.new('RGB', (640, 480), color='white')
draw = ImageDraw.Draw(img)

# Vẽ vài hình để YOLO có thể detect
draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=3)  # Square
draw.rectangle([300, 150, 400, 300], fill='blue', outline='black', width=3)  # Rectangle
draw.ellipse([450, 200, 550, 350], fill='green', outline='black', width=3)  # Circle

# Lưu ảnh
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
image_bytes = buffer.getvalue()

# Gửi lên API
files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
data = {'confidence_threshold': 0.1}

print("🚀 Testing YOLOv8 Detection...")
print("📤 Sending test image to API...")

try:
    response = requests.post('http://127.0.0.1:8000/api/detect-upload', files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Found {result['total_detections']} objects")
        print(f"⏱️  Processing time: {result['processing_time']:.4f}s")
        print(f"🎯 Detected classes: {result['detected_classes']}")
        
        print("\n📋 Detection Details:")
        for i, det in enumerate(result['detections']):
            print(f"  {i+1}. {det['class_name']} (confidence: {det['confidence']:.3f})")
            print(f"     Bbox: ({det['bbox']['x_min']}, {det['bbox']['y_min']}) → ({det['bbox']['x_max']}, {det['bbox']['y_max']})")
            print(f"     Color: {det['color']}")
        
        # Kiểm tra annotated image
        if result.get('annotated_image'):
            print(f"🖼️  Annotated image available: {len(result['annotated_image'])} chars")
        else:
            print("❌ No annotated image returned")
            
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")

print("\n🎯 Test completed!")
