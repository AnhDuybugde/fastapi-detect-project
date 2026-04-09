# Simplified FastAPI - Only Lightweight Detection
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import time

try:
    from lightweight_detector import lightweight_detector, detect_fruits_in_image as lw_detect
except ImportError:
    pass

try:
    from yolo_detector import yolo_detector
    AI_AVAILABLE = True
    # Pre-load model so /api/fruits has classes ready immediately
    yolo_detector.load_model()
except ImportError:
    AI_AVAILABLE = False

def detect_fruits_in_image(image_bytes: bytes, conf_threshold: float = 0.5):
    if AI_AVAILABLE and yolo_detector.model is not None:
        return yolo_detector.detect_image(image_bytes, conf_threshold)
    elif AI_AVAILABLE:
        # Load the model the first time
        yolo_detector.load_model()
        if yolo_detector.model is not None:
             return yolo_detector.detect_image(image_bytes, conf_threshold)
             
    # Fallback to lightweight
    return lw_detect(image_bytes)

app = FastAPI(title="YOLO-World Fruit Detection API", version="4.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int

class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: BoundingBox
    color: str

class DetectionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    detections: List[Detection]
    total_detections: int
    detected_classes: List[str]
    model_info: dict
    processing_time: float
    image_info: dict
    image_size: Optional[dict] = None
    annotated_image: Optional[str] = None

# Endpoints
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": AI_AVAILABLE and yolo_detector.model is not None,
        "timestamp": time.time(),
        "mode": "yolo_world" if AI_AVAILABLE else "lightweight_fallback"
    }

@app.get("/api/fruits")
async def get_fruits():
    """Get list of detectable fruit classes."""
    if AI_AVAILABLE:
        return {"classes": yolo_detector.classes}
    else:
        return {"classes": list(lightweight_detector.fruit_classes.keys())}

@app.post("/api/detect-upload", response_model=DetectionResponse)
async def detect_fruits_upload(
    image: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(0.25, description="Minimum confidence threshold"),
    return_annotated: Optional[bool] = Form(True, description="Return image with bounding boxes drawn"),
):
    """
    Detect fruits in uploaded image using lightweight detection only.
    Perfect for serverless deployments (Vercel, Render).
    """
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File phải là ảnh (image/*)")

    # Validate file size
    image_data = await image.read()
    if len(image_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File ảnh quá lớn (tối đa 10MB)")

    try:
        start_time = time.time()

        # Dùng model với biến conf
        result = detect_fruits_in_image(image_data, confidence_threshold)

        processing_time = time.time() - start_time

        # Lọc theo confidence threshold
        filtered_detections = [
            d for d in result['detections']
            if d['confidence'] >= confidence_threshold
        ]

        # Map to response format
        detections_out = [
            Detection(
                class_name=d.get('class_name', d.get('class', 'unknown')),
                class_id=d['class_id'],
                confidence=d['confidence'],
                bbox=BoundingBox(**d['bbox']),
                color=d['color'],
            )
            for d in filtered_detections
        ]

        # Vẽ bounding boxes nếu cần
        annotated_image = None
        if return_annotated:
            if AI_AVAILABLE and yolo_detector.model is not None:
                annotated_image = yolo_detector.draw_detections(image_data, filtered_detections)
            else:
                annotated_image = lightweight_detector.draw_detections(image_data, filtered_detections)

        detected_classes = list(set(d.get('class', d.get('class_name', 'unknown')) for d in filtered_detections))

        return DetectionResponse(
            detections=detections_out,
            total_detections=len(detections_out),
            detected_classes=detected_classes,
            model_info=result['model_info'],
            processing_time=round(processing_time, 4),
            image_info={
                "filename": image.filename,
                "size_bytes": len(image_data),
                "content_type": image.content_type,
            },
            image_size=result.get('image_size'),
            annotated_image=annotated_image,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection thất bại: {str(e)}")

@app.get("/api/stats")
async def get_detection_stats():
    """Get detection statistics and model info."""
    model_info = {}
    if AI_AVAILABLE:
        model_info = yolo_detector.get_info()
    else:
        model_info = lightweight_detector.get_info()
        
    return {
        "model_info": model_info,
        "api_endpoints": [
            "GET  /api/health          — Health check",
            "GET  /api/fruits          — Available fruit classes",
            "POST /api/detect-upload   — Upload ảnh để detect (Lightweight Only)",
            "GET  /api/stats           — Endpoint này",
        ],
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "webp"],
        "max_file_size":      "10MB",
        "mode":              "lightweight_only",
        "deployment_target":   "serverless_optimized",
    }

@app.get("/")
async def root():
    return {
        "message":     "YOLO-World Fruit Detection API v4.0",
        "description": "AI-powered fruit detection using YOLO-World open-vocabulary model",
        "mode":        "yolo_world" if AI_AVAILABLE else "lightweight_fallback",
        "endpoints": {
            "health":         "/api/health",
            "fruits":         "/api/fruits",
            "detect_upload":  "/api/detect-upload",
            "stats":          "/api/stats",
            "docs":           "/docs",
        },
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Khởi động YOLO-World Fruit Detection API tại http://127.0.0.1:8000")
    print("📖 API Documentation: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
