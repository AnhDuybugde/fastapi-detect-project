from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Maritime Detection API", version="1.0")

# Thêm đoạn này để mở khóa CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả các web gọi tới
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Điểm nổi bật 1: Pydantic Model. 
# Định nghĩa CHÍNH XÁC cấu trúc dữ liệu đầu vào/đầu ra. Sai kiểu dữ liệu = tự động báo lỗi.
class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    object_class: str

class DetectionResponse(BaseModel):
    image_id: str
    detections: List[BoundingBox]

# Điểm nổi bật 2: 'async def' giúp API không bị "đóng băng" khi chờ mô hình chạy
@app.post("/api/detect", response_model=DetectionResponse)
async def run_detection(image_id: str):
    if len(image_id) < 3:
        # Xử lý lỗi gọn gàng hơn Flask
        raise HTTPException(status_code=400, detail="Image ID quá ngắn")
    
    # Giả lập kết quả từ mô hình Computer Vision (tìm thấy 1 người trên biển)
    mock_bbox = BoundingBox(
        x_min=120, y_min=45, x_max=200, y_max=310, 
        confidence=0.95, object_class="person_in_water"
    )
    
    return DetectionResponse(image_id=image_id, detections=[mock_bbox])