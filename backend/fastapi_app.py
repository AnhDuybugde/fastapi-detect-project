# Simplified FastAPI - Only Lightweight Detection
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time

# Only import lightweight detector
from lightweight_detector import detect_fruits_in_image, get_available_fruits

app = FastAPI(title="Lightweight Fruit Detection API", version="3.0")

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
    detections: List[Detection]
    total_detections: int
    detected_classes: List[str]
    model_info: dict
    processing_time: float
    image_info: dict
    annotated_image: Optional[str] = None

# Endpoints
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": time.time(),
        "mode": "lightweight_only"
    }

@app.get("/api/fruits")
async def get_fruits():
    """Get list of detectable fruit classes."""
    return get_available_fruits()

@app.post("/api/detect-upload", response_model=DetectionResponse)
async def detect_fruits_upload(
    image: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(0.5, description="Minimum confidence threshold"),
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

        # Simple: chỉ dùng lightweight detection
        result = detect_fruits_in_image(image_data)

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
            from lightweight_detector import lightweight_detector
            annotated_image = lightweight_detector.draw_detections(image_data, filtered_detections)

        detected_classes = list(set(d['class'] for d in filtered_detections))

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
            annotated_image=annotated_image,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection thất bại: {str(e)}")

@app.get("/api/stats")
async def get_detection_stats():
    """Get detection statistics and model info."""
    return {
        "model_info": get_available_fruits(),
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
        "message":     "Lightweight Fruit Detection API v3.0",
        "description": "Fast, serverless-optimized fruit detection",
        "mode":        "lightweight_only",
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
    print("Khởi động Lightweight FastAPI Server tại http://127.0.0.1:8000")
    print("API Documentation: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
