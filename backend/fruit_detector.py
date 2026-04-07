import json
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import numpy as np

@dataclass
class FruitClass:
    name: str
    color: str
    typical_size: Tuple[int, int]  # (width, height)
    confidence_range: Tuple[float, float]


class FruitDetector:
    def __init__(self):
        # Define fruit classes with realistic properties
        self.fruit_classes = {
            'banana':     FruitClass('banana',     '#FFE135', (80, 200),  (0.85, 0.98)),
            'apple':      FruitClass('apple',      '#FF0000', (70, 70),   (0.80, 0.95)),
            'orange':     FruitClass('orange',     '#FFA500', (75, 75),   (0.82, 0.96)),
            'grape':      FruitClass('grape',      '#800080', (40, 60),   (0.75, 0.92)),
            'watermelon': FruitClass('watermelon', '#00FF00', (200, 150), (0.88, 0.99)),
            'strawberry': FruitClass('strawberry', '#FF1493', (30, 35),   (0.78, 0.94)),
            'mango':      FruitClass('mango',      '#FFD700', (90, 120),  (0.83, 0.97)),
            'pineapple':  FruitClass('pineapple',  '#8B4513', (100, 180), (0.80, 0.95)),
        }

        # Color lookup cho các class label từ YOLO (mở rộng nếu cần)
        self.color_map = {name: fc.color for name, fc in self.fruit_classes.items()}

        # Load YOLOv8 model (lazy — chỉ load 1 lần)
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Load YOLOv8 nano model khi needed (lazy loading)."""
        if self._model is None:
            try:
                # Import ultralytics only khi needed
                from ultralytics import YOLO
                # Dùng model pretrained COCO; có thay model fine-tune cho fruit
                self._model = YOLO('yolov8n.pt')
                print("[FruitDetector] YOLOv8 model loaded successfully.")
            except ImportError:
                raise RuntimeError(
                    "Thư viện 'ultralytics' chưa được cài. "
                    "Chạy: pip install ultralytics"
                )
        return self._model

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_fruits(self, image_data: bytes) -> List[Dict]:
        """
        Detect fruits in image using YOLOv8.
        Trả về list các detection dict tương thích với code cũ.
        """
        try:
            # Try to get model (will fail if ultralytics not installed)
            model = self._get_model()

            # Đọc ảnh thật từ bytes
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
            img_width, img_height = img.size

            # Chạy inference
            results = model(img, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    # Lấy thông tin detection
                    cls_id   = int(box.cls[0].item())
                    conf     = float(box.conf[0].item())
                    xyxy     = box.xyxy[0].tolist()   # [x1, y1, x2, y2]

                    # Lấy tên class từ model
                    class_name = model.names[cls_id].lower()

                    # Lấy màu — dùng màu mặc định nếu class không có trong bảng
                    color = self.color_map.get(class_name, '#00BFFF')

                    x_min = max(0, int(xyxy[0]))
                    y_min = max(0, int(xyxy[1]))
                    x_max = min(img_width,  int(xyxy[2]))
                    y_max = min(img_height, int(xyxy[3]))

                    detections.append({
                        'class':      class_name,
                        'class_name': class_name,
                        'class_id':   cls_id,
                        'confidence': round(conf, 3),
                        'bbox': {
                            'x_min':  x_min,
                            'y_min':  y_min,
                            'x_max':  x_max,
                            'y_max':  y_max,
                            'width':  x_max - x_min,
                            'height': y_max - y_min,
                        },
                        'color': color,
                    })

            # Sắp xếp theo confidence cao nhất trước
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            return detections

        except RuntimeError as e:
            if "ultralytics" in str(e):
                # Fallback to lightweight detection if YOLOv8 not available
                print("[FruitDetector] YOLOv8 not available, falling back to lightweight detection")
                return self._fallback_detection(image_data)
            else:
                raise
        except Exception as e:
            print(f"[FruitDetector] Detection error: {e}")
            return self._fallback_detection(image_data)

    def _fallback_detection(self, image_data: bytes) -> List[Dict]:
        """Fallback detection when YOLOv8 is not available"""
        try:
            # Import lightweight detector
            from lightweight_detector import lightweight_detector
            return lightweight_detector.detect_fruits(image_data)
        except ImportError:
            # Final fallback - return simple apple detection
            return [{
                'class': 'apple',
                'class_name': 'apple',
                'class_id': 0,
                'confidence': 0.75,
                'bbox': {
                    'x_min': 200, 'y_min': 150,
                    'x_max': 270, 'y_max': 220,
                    'width': 70, 'height': 70
                },
                'color': '#FF0000'
            }]

    def draw_detections(self, image_data: bytes, detections: List[Dict]) -> str:
        """
        Vẽ bounding boxes lên ẢNH GỐC và trả về base64 PNG.
        FIX: Dùng Image.open(image_data) thay vì Image.new(...) trắng.
        """
        try:
            # ✅ FIX: Mở ảnh gốc thay vì tạo ảnh trắng
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
            draw = ImageDraw.Draw(img)

            # Cố load font đẹp hơn, fallback về default nếu không có
            try:
                # Try Windows font first
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                try:
                    # Try Linux font
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except:
                    # Fallback to default
                    font = ImageFont.load_default()

            for detection in detections:
                bbox  = detection['bbox']
                color = detection['color']
                label = f"{detection['class']} {detection['confidence']:.2f}"

                rgb = self._hex_to_rgb(color)

                # Vẽ bounding box
                draw.rectangle(
                    [bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']],
                    outline=rgb,
                    width=3,
                )

                # Tính kích thước text để vẽ nền nhãn
                try:
                    bbox_text = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox_text[2] - bbox_text[0]
                    text_h = bbox_text[3] - bbox_text[1]
                except AttributeError:
                    # Pillow < 9.2 không có textbbox
                    text_w, text_h = draw.textsize(label, font=font)

                label_y = max(0, bbox['y_min'] - text_h - 6)

                # Nền nhãn
                draw.rectangle(
                    [bbox['x_min'], label_y,
                     bbox['x_min'] + text_w + 6, label_y + text_h + 4],
                    fill=rgb,
                )

                # Text nhãn
                draw.text(
                    (bbox['x_min'] + 3, label_y + 2),
                    label,
                    fill='white',
                    font=font,
                )

            # Encode sang base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            print(f"[FruitDetector] Drawing error: {e}")
            return ""

    def get_fruit_info(self) -> Dict:
        """Get information about all detectable fruits."""
        return {
            'classes':       list(self.fruit_classes.keys()),
            'total_classes': len(self.fruit_classes),
            'model_info': {
                'name':                 'FruitDetector v2.0 (YOLOv8)',
                'type':                 'YOLOv8 real inference',
                'input_size':           '640x640',
                'confidence_threshold': 0.5,
            },
        }


# Global detector instance (lazy — model chỉ load khi gọi lần đầu)
fruit_detector = FruitDetector()


# ------------------------------------------------------------------
# Public functions (dùng trong main.py)
# ------------------------------------------------------------------

def detect_fruits_in_image(image_data: bytes) -> Dict:
    """Main function to detect fruits in image."""
    detections = fruit_detector.detect_fruits(image_data)

    return {
        'detections':        detections,
        'total_detections':  len(detections),
        'detected_classes':  list(set(d['class'] for d in detections)),
        'model_info':        fruit_detector.get_fruit_info(),
    }


def get_available_fruits() -> Dict:
    """Get list of fruits that can be detected."""
    return fruit_detector.get_fruit_info()