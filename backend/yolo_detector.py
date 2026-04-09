"""
Hybrid Fruit Detector v3 — Best of Both Worlds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy:
  • Custom best.pt (6 classes) → Trusted for: pineapple, cherry, mango,
                                    plum, tomato, watermelon
  • YOLO-World (zero-shot)    → Fills the gaps: banana, apple, orange,
                                    grape, strawberry

Logic:
  1. Run BOTH models on the image
  2. For classes best.pt knows → trust best.pt (higher accuracy)
  3. For classes best.pt doesn't know → use YOLO-World
  4. Merge results, de-duplicate overlapping boxes via IoU
"""
import time
import io
import os
import base64
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Lazy load ultralytics
try:
    from ultralytics import YOLOWorld, YOLO
    YOLO_AVAILABLE = True
except ImportError:
    try:
        from ultralytics import YOLO
        YOLOWorld = YOLO
        YOLO_AVAILABLE = True
    except ImportError:
        YOLO_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
#  Class definitions
# ═══════════════════════════════════════════════════════════════

# The 8 classes we want to detect (canonical order)
ALL_FRUIT_CLASSES = [
    "banana", "apple", "orange", "grape",
    "watermelon", "strawberry", "mango", "pineapple",
]

# Classes the custom best.pt model knows (from classes.json)
CUSTOM_MODEL_CLASSES = {
    0: "pineapple",
    1: "cherry",
    2: "mango",
    3: "plum",
    4: "tomato",
    5: "watermelon",
}

# Which of our 8 target classes does best.pt handle?
CUSTOM_HANDLES = {"pineapple", "mango", "watermelon"}

# Which classes need YOLO-World to fill the gap?
WORLD_HANDLES = {"banana", "apple", "orange", "grape", "strawberry"}

# Descriptive prompts for YOLO-World (prompt-engineered)
WORLD_PROMPTS = [
    "yellow banana fruit",
    "red apple fruit",
    "round orange citrus fruit",
    "bunch of purple grapes",
    "small red strawberry with seeds",
]
WORLD_PROMPT_MAP = ["banana", "apple", "orange", "grape", "strawberry"]

# Hex colours for bounding boxes
CLASS_COLORS = {
    "banana":      "#FFE135",
    "apple":       "#FF0000",
    "orange":      "#FFA500",
    "grape":       "#800080",
    "watermelon":  "#00FF00",
    "strawberry":  "#FF1493",
    "mango":       "#FFD700",
    "pineapple":   "#8B4513",
}


def _iou(box_a: Dict, box_b: Dict) -> float:
    """Calculate IoU between two bounding boxes."""
    x1 = max(box_a["x_min"], box_b["x_min"])
    y1 = max(box_a["y_min"], box_b["y_min"])
    x2 = min(box_a["x_max"], box_b["x_max"])
    y2 = min(box_a["y_max"], box_b["y_max"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = box_a["width"] * box_a["height"]
    area_b = box_b["width"] * box_b["height"]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0

def _white_balance(img_crop: np.ndarray) -> np.ndarray:
    """Apply vectorized Gray World White Balance to handle lighting issues."""
    avg_color = img_crop.mean(axis=(0, 1))
    avg_r, avg_g, avg_b = avg_color[0], avg_color[1], avg_color[2]
    
    k = (avg_r + avg_g + avg_b) / 3.0
    if avg_r > 0 and avg_g > 0 and avg_b > 0:
        # Vectorized scaling using NumPy broadcasting
        alpha = np.array([k/avg_r, k/avg_g, k/avg_b], dtype=np.float32)
        balanced = np.clip(img_crop * alpha, 0, 255).astype(np.uint8)
        return balanced
    return img_crop

def _apply_clahe(img_np: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to balance exposure."""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    merged = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


class YoloFruitDetector:
    def __init__(self):
        self.custom_model = None       # best.pt  (6 classes)
        self.world_model = None        # YOLO-World (open vocab)
        self.classes = list(ALL_FRUIT_CLASSES)
        self.class_colors = dict(CLASS_COLORS)
        self.model = True              # Compatibility flag for fastapi_app.py

    # ── Model loading ───────────────────────────────────────────
    def load_model(self):
        """Load both models for hybrid detection."""
        if not YOLO_AVAILABLE:
            print("❌ ultralytics not installed")
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Load custom best.pt
        if self.custom_model is None:
            best_path = os.path.join(base_dir, "weights", "best.pt")
            if os.path.exists(best_path):
                try:
                    print(f"[Hybrid] Loading custom model: {best_path}")
                    self.custom_model = YOLO(best_path)
                    print(f"  ✅ Custom model loaded — handles: {CUSTOM_HANDLES}")
                except Exception as e:
                    print(f"  ❌ Custom model failed: {e}")

        # 2. Load YOLO-World (Strategy 1: Model Scaling -> use medium model if available)
        if self.world_model is None:
            world_path_m = os.path.join(base_dir, "weights", "yolov8m-world.pt")
            world_path_s = os.path.join(base_dir, "weights", "yolov8s-world.pt")
            world_path = world_path_m if os.path.exists(world_path_m) else world_path_s
            
            if os.path.exists(world_path):
                try:
                    print(f"[Hybrid] Loading YOLO-World: {world_path}")
                    self.world_model = YOLOWorld(world_path)
                    self.world_model.set_classes(WORLD_PROMPTS)
                    print(f"  ✅ YOLO-World loaded — handles: {WORLD_HANDLES}")
                except Exception as e:
                    print(f"  ❌ YOLO-World failed: {e}")

        self.model = (self.custom_model is not None or self.world_model is not None)
        mode = []
        if self.custom_model:
            mode.append("Custom(pineapple,mango,watermelon)")
        if self.world_model:
            mode.append("World(banana,apple,orange,grape,strawberry)")
        print(f"  🔀 Hybrid mode: {' + '.join(mode)}")

    # ── Info ─────────────────────────────────────────────────────
    def get_info(self) -> Dict[str, Any]:
        return {
            "name":        "Hybrid Fruit Detector v3",
            "model_type":  "Custom YOLOv8 + YOLO-World (Hybrid)",
            "classes":     len(self.classes),
            "class_list":  self.classes,
            "available":   bool(self.model),
            "type":        "Hybrid AI detection (best of both worlds)",
            "description": "Custom model for trained classes + YOLO-World for the rest",
        }

    # ── Detection ────────────────────────────────────────────────
    def detect_image(self, image_bytes: bytes, conf_threshold: float = 0.5) -> Dict[str, Any]:
        """Run hybrid detection — merge results from both models."""
        start_time = time.time()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        
        # Thử thách 2: Dynamic Exposure - Cân bằng sáng toàn khung hình trước khi nhận diện
        img_np_raw = np.array(img)
        img_clahe = _apply_clahe(img_np_raw)
        img = Image.fromarray(img_clahe)

        if not self.model:
            self.load_model()

        all_detections: List[Dict] = []

        # ── Run Custom model (best.pt) ──
        if self.custom_model:
            try:
                # Strategy 2: Conf Threshold passed dynamically
                results = self.custom_model.predict(img, conf=conf_threshold, iou=0.45, verbose=False)
                for result in results:
                    for box in result.boxes:
                        b = box.xyxy[0].tolist()
                        c = int(box.cls)
                        conf = float(box.conf)
                        raw_name = CUSTOM_MODEL_CLASSES.get(c, f"class_{c}")
                        # Only keep classes we care about
                        if raw_name in CUSTOM_HANDLES:
                            all_detections.append(self._make_detection(
                                raw_name, c, conf, b, source="custom"
                            ))
            except Exception as e:
                print(f"[Hybrid] Custom model error: {e}")

        # ── Run YOLO-World ──
        if self.world_model:
            try:
                # Strategy 2: Conf Threshold passed dynamically
                results = self.world_model.predict(img, conf=conf_threshold, iou=0.45, verbose=False)
                for result in results:
                    for box in result.boxes:
                        b = box.xyxy[0].tolist()
                        c = int(box.cls)
                        conf = float(box.conf)
                        if c < len(WORLD_PROMPT_MAP):
                            class_name = WORLD_PROMPT_MAP[c]
                        else:
                            class_name = f"class_{c}"
                        # Only keep classes YOLO-World should handle
                        if class_name in WORLD_HANDLES:
                            
                            # Strategy 4: Logic hậu kỳ (Post-processing check based on color)
                            # Kiểm tra màu sắc nếu nhận diện là 'orange'
                            if class_name == "orange":
                                x1, y1, x2, y2 = map(int, b)
                                img_np = np.array(img)
                                crop = img_np[max(0, y1):y2, max(0, x1):x2]
                                if crop.shape[0] > 0 and crop.shape[1] > 0:
                                    # Cân bằng trắng (White Balance) trước khi đo màu để tránh ám màu ánh sáng
                                    crop_wb = _white_balance(crop)
                                    hsv = cv2.cvtColor(crop_wb, cv2.COLOR_RGB2HSV)
                                    h_median = np.median(hsv[:, :, 0])
                                    # OpenCV Hue (0-179). Màu xanh dải khoảng 35-85.
                                    # Nếu nó có màu xanh đậm (Cam sành -> hoặc có thể là ngụy trang)
                                    if 35 < h_median < 85:
                                        print(f"  [Filter] Đã lọc bỏ nhận diện sai 'orange' do màu hue xanh đậm ({h_median})")
                                        continue
                                        
                            all_detections.append(self._make_detection(
                                class_name, ALL_FRUIT_CLASSES.index(class_name),
                                conf, b, source="world"
                            ))
            except Exception as e:
                print(f"[Hybrid] YOLO-World error: {e}")

        # ── De-duplicate overlapping boxes ──
        all_detections = self._deduplicate(all_detections)

        # Sort by confidence
        all_detections.sort(key=lambda d: d["confidence"], reverse=True)

        processing_time = time.time() - start_time

        return {
            "detections":      all_detections,
            "processing_time": processing_time,
            "image_size":      {"width": width, "height": height},
            "model_info":      self.get_info(),
        }

    def _make_detection(self, class_name: str, class_id: int, conf: float,
                        xyxy: list, source: str = "") -> Dict:
        """Create a standardized detection dict."""
        color = self.class_colors.get(class_name, "#FFFFFF")
        return {
            "class_id":   class_id,
            "class":      class_name,
            "class_name": class_name.capitalize(),
            "confidence": round(conf, 4),
            "color":      color,
            "bbox": {
                "x_min":  int(xyxy[0]),
                "y_min":  int(xyxy[1]),
                "x_max":  int(xyxy[2]),
                "y_max":  int(xyxy[3]),
                "width":  int(xyxy[2] - xyxy[0]),
                "height": int(xyxy[3] - xyxy[1]),
            },
            "_source": source,  # internal: which model produced this
        }

    def _deduplicate(self, detections: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
        """Remove duplicate detections where boxes overlap significantly."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        keep = []

        for det in detections:
            is_dup = False
            for kept in keep:
                if _iou(det["bbox"], kept["bbox"]) > iou_thresh:
                    # Same region — keep the one with higher confidence (already sorted)
                    is_dup = True
                    break
            if not is_dup:
                keep.append(det)

        return keep

    # ── Draw bounding boxes ──────────────────────────────────────
    def draw_detections(self, image_bytes: bytes, detections: List[Dict]) -> str:
        """Draw bounding boxes and return base64-encoded image."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        img_w, img_h = img.size
        font_size = max(14, min(28, img_w // 30))

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for d in detections:
            bbox = d["bbox"]
            conf = d["confidence"]
            name = d.get("class_name", d.get("class", "?"))
            color = d.get("color", "#00FF00")

            shape = [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]
            line_width = max(2, img_w // 200)
            draw.rectangle(shape, outline=color, width=line_width)

            label = f"{name} {conf:.1%}"

            if hasattr(font, "getbbox"):
                txt_bbox = font.getbbox(label)
                tw = txt_bbox[2] - txt_bbox[0]
                th = txt_bbox[3] - txt_bbox[1]
            else:
                tw, th = draw.textsize(label, font=font)

            label_y = max(0, shape[1] - th - 8)
            draw.rectangle(
                [shape[0], label_y, shape[0] + tw + 8, label_y + th + 6],
                fill=color,
            )
            draw.text((shape[0] + 4, label_y + 3), label, fill="white", font=font)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ── Singleton ────────────────────────────────────────────────────
yolo_detector = YoloFruitDetector()
