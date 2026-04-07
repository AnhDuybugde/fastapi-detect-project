from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import uuid
import time
from fruit_detector import detect_fruits_in_image, get_available_fruits

app = FastAPI(title="Fruit Detection API", version="2.0")

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả các web gọi tới
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

# Health check
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": time.time()
    }

# Get available fruit classes
@app.get("/api/fruits")
async def get_fruits():
    """Get list of detectable fruit classes"""
    return get_available_fruits()

# Original endpoint for backward compatibility
@app.post("/api/detect", response_model=DetectionResponse)
async def run_detection(image_id: str):
    """Legacy endpoint for backward compatibility"""
    if len(image_id) < 3:
        raise HTTPException(status_code=400, detail="Image ID quá ngắn")
    
    # Simulate detection with mock data
    mock_detection = Detection(
        class_name="banana",
        class_id=0,
        confidence=0.95,
        bbox=BoundingBox(x_min=120, y_min=45, x_max=200, y_max=310, width=80, height=265),
        color="#FFE135"
    )
    
    return DetectionResponse(
        detections=[mock_detection],
        total_detections=1,
        detected_classes=["banana"],
        model_info={"name": "FruitDetector v2.0", "type": "YOLO-based"},
        processing_time=0.05,
        image_info={"id": image_id, "size": "640x480"}
    )

# New image upload endpoint
@app.post("/api/detect-upload", response_model=DetectionResponse)
async def detect_fruits_upload(
    image: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(0.5, description="Minimum confidence threshold")
):
    """
    Detect fruits in uploaded image
    """
    # Validate file
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if image.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
    
    try:
        # Read image data
        image_data = await image.read()
        
        # Start timing
        start_time = time.time()
        
        # Perform detection
        result = detect_fruits_in_image(image_data)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Convert to response format
        detections = []
        for det in result['detections']:
            if det['confidence'] >= confidence_threshold:
                detections.append(Detection(
                    class_name=det['class'],
                    class_id=det['class_id'],
                    confidence=det['confidence'],
                    bbox=BoundingBox(**det['bbox']),
                    color=det['color']
                ))
        
        return DetectionResponse(
            detections=detections,
            total_detections=len(detections),
            detected_classes=result['detected_classes'],
            model_info=result['model_info'],
            processing_time=processing_time,
            image_info={
                "filename": image.filename,
                "size": f"{len(image_data)} bytes",
                "content_type": image.content_type
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# Batch detection endpoint
@app.post("/api/detect-batch")
async def detect_fruits_batch(
    images: List[UploadFile] = File(..., description="Multiple image files to analyze")
):
    """
    Detect fruits in multiple images
    """
    if len(images) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many images (max 10)")
    
    results = []
    for i, image in enumerate(images):
        try:
            image_data = await image.read()
            result = detect_fruits_in_image(image_data)
            
            results.append({
                "image_index": i,
                "filename": image.filename,
                "detections": result['detections'],
                "total_detections": result['total_detections'],
                "detected_classes": result['detected_classes']
            })
            
        except Exception as e:
            results.append({
                "image_index": i,
                "filename": image.filename,
                "error": str(e),
                "detections": [],
                "total_detections": 0,
                "detected_classes": []
            })
    
    return {
        "batch_results": results,
        "total_images": len(images),
        "successful_detections": sum(1 for r in results if "error" not in r)
    }

# Statistics endpoint
@app.get("/api/stats")
async def get_detection_stats():
    """Get detection statistics and model info"""
    return {
        "model_info": get_available_fruits(),
        "api_endpoints": [
            "GET /api/health - Health check",
            "GET /api/fruits - Available fruit classes", 
            "POST /api/detect - Legacy endpoint",
            "POST /api/detect-upload - Upload image for detection",
            "POST /api/detect-batch - Batch detection",
            "GET /api/stats - This statistics endpoint"
        ],
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "webp"],
        "max_file_size": "10MB",
        "max_batch_size": 10
    }

# Root endpoint with API documentation
@app.get("/")
async def root():
    return {
        "message": "Fruit Detection API v2.0",
        "description": "Advanced fruit detection using YOLO-based model",
        "endpoints": {
            "health": "/api/health",
            "fruits": "/api/fruits", 
            "detect_upload": "/api/detect-upload",
            "detect_batch": "/api/detect-batch",
            "stats": "/api/stats",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Khởi động FastAPI Server tại http://127.0.0.1:8000")
    print("API Documentation: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)