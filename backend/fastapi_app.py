from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import time

from fruit_detector import detect_fruits_in_image, get_available_fruits, fruit_detector
from lightweight_detector import detect_fruits_in_image as lightweight_detect, get_available_fruits as lightweight_get_fruits

app = FastAPI(title="Fruit Detection API", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int


class Detection(BaseModel):
    # ✅ FIX: thống nhất dùng 'class_name' xuyên suốt
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
    # ✅ THÊM MỚI: ảnh gốc đã được vẽ bbox (base64 PNG)
    annotated_image: Optional[str] = None


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {
        "status":       "healthy",
        "model_loaded": True,
        "timestamp":    time.time(),
    }


@app.get("/api/fruits")
async def get_fruits():
    """Get list of detectable fruit classes."""
    return get_available_fruits()


@app.post("/api/detect", response_model=DetectionResponse)
async def run_detection(image_id: str):
    """Legacy endpoint for backward compatibility."""
    if len(image_id) < 3:
        raise HTTPException(status_code=400, detail="Image ID quá ngắn")

    mock_detection = Detection(
        class_name="banana",
        class_id=0,
        confidence=0.95,
        bbox=BoundingBox(x_min=120, y_min=45, x_max=200, y_max=310, width=80, height=265),
        color="#FFE135",
    )

    return DetectionResponse(
        detections=[mock_detection],
        total_detections=1,
        detected_classes=["banana"],
        model_info={"name": "FruitDetector v2.0", "type": "YOLOv8"},
        processing_time=0.05,
        image_info={"id": image_id, "size": "640x480"},
        annotated_image=None,
    )


@app.post("/api/detect-upload", response_model=DetectionResponse)
async def detect_fruits_upload(
    image: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(0.5, description="Minimum confidence threshold"),
    return_annotated: Optional[bool] = Form(True, description="Return image with bounding boxes drawn"),
    use_lightweight: Optional[bool] = Form(False, description="Use lightweight detection (faster for serverless)"),
):
    """
    Detect fruits in uploaded image.
    Trả về JSON detections + (tuỳ chọn) ảnh đã vẽ bbox dạng base64.
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

        # ✅ Choose detection mode based on parameter
        if use_lightweight:
            # Lightweight detection for Vercel/Render
            result = lightweight_detect(image_data)
        else:
            # Full YOLOv8 detection
            result = detect_fruits_in_image(image_data)

        processing_time = time.time() - start_time

        # Lọc theo confidence threshold
        filtered_detections = [
            d for d in result['detections']
            if d['confidence'] >= confidence_threshold
        ]

        # ✅ FIX: Map đúng key 'class' → 'class_name'
        detections_out = [
            Detection(
                class_name=d.get('class_name', d.get('class', 'unknown')),  # Handle both fields
                class_id=d['class_id'],
                confidence=d['confidence'],
                bbox=BoundingBox(**d['bbox']),
                color=d['color'],
            )
            for d in filtered_detections
        ]

        # ✅ THÊM MỚI: Vẽ bbox lên ảnh gốc (nếu yêu cầu)
        annotated_image = None
        if return_annotated:
            annotated_image = fruit_detector.draw_detections(image_data, filtered_detections)

        # Detected classes chỉ lấy từ những detection đã pass threshold
        detected_classes = list(set(d['class'] for d in filtered_detections))

        return DetectionResponse(
            detections=detections_out,
            total_detections=len(detections_out),
            detected_classes=detected_classes,
            model_info=result['model_info'],
            processing_time=round(processing_time, 4),
            image_info={
                "filename":     image.filename,
                "size_bytes":   len(image_data),
                "content_type": image.content_type,
            },
            annotated_image=annotated_image,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection thất bại: {str(e)}")


@app.post("/api/detect-batch")
async def detect_fruits_batch(
    images: List[UploadFile] = File(..., description="Multiple image files to analyze"),
    confidence_threshold: Optional[float] = Form(0.5),
):
    """Detect fruits in multiple images (max 10)."""
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Quá nhiều ảnh (tối đa 10)")

    results = []
    for i, image in enumerate(images):
        try:
            image_data = await image.read()
            result = detect_fruits_in_image(image_data)

            # Lọc theo threshold
            filtered = [d for d in result['detections'] if d['confidence'] >= confidence_threshold]

            results.append({
                "image_index":      i,
                "filename":         image.filename,
                "detections":       filtered,
                "total_detections": len(filtered),
                "detected_classes": list(set(d['class'] for d in filtered)),
            })

        except Exception as e:
            results.append({
                "image_index":      i,
                "filename":         image.filename,
                "error":            str(e),
                "detections":       [],
                "total_detections": 0,
                "detected_classes": [],
            })

    return {
        "batch_results":          results,
        "total_images":           len(images),
        "successful_detections":  sum(1 for r in results if "error" not in r),
    }


@app.get("/api/stats")
async def get_detection_stats():
    """Get detection statistics and model info."""
    return {
        "model_info": get_available_fruits(),
        "api_endpoints": [
            "GET  /api/health          — Health check",
            "GET  /api/fruits          — Available fruit classes",
            "POST /api/detect          — Legacy endpoint (mock)",
            "POST /api/detect-upload   — Upload ảnh để detect (YOLOv8)",
            "POST /api/detect-batch    — Batch detection",
            "GET  /api/stats           — Endpoint này",
        ],
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "webp"],
        "max_file_size":      "10MB",
        "max_batch_size":     10,
    }


@app.get("/")
async def root():
    return {
        "message":     "Fruit Detection API v2.0",
        "description": "Real fruit detection using YOLOv8",
        "endpoints": {
            "health":         "/api/health",
            "fruits":         "/api/fruits",
            "detect_upload":  "/api/detect-upload",
            "detect_batch":   "/api/detect-batch",
            "stats":          "/api/stats",
            "docs":           "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    print("Khởi động FastAPI Server tại http://127.0.0.1:8000")
    print("API Documentation: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)