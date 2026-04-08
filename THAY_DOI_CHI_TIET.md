# CHI TIẾT CÁC THAY ĐỔI VẬT LÝ (BẢN V3 HYBRID SO VỚI BẢN CŨ)

Tất cả các file cập nhật hiện đang nằm trực tiếp tại thư mục dự án của bạn (`d:\Triet\fastapi-detect-project-main`), tôi đã gỡ bỏ nhánh Git ảo để mọi thứ trở về dạng thư mục vật lý theo ý bạn.

Dưới đây là chi tiết từng file một đã được sửa đổi và mục đích của nó:

### 📁 Thư mục `backend/`

1. **`backend/yolo_detector.py` (GHI ĐÈ HOÀN TOÀN TỪ CŨ SANG MỚI)**
   - **Bản cũ:** Chỉ tải mỗi file `best.pt` của bạn, với 6 nhãn (thiếu class).
   - **Đã thay đổi:** Viết lại toàn bộ bằng Cơ chế Hybrid mạnh mẽ:
     - Gọi `ultralytics.YOLO` tải `best.pt` để nhận diện độc quyền: Dưa hấu, Xoài, Dứa (vì đây là các quả bản cũ của bạn làm tốt).
     - Gọi `ultralytics.YOLOWorld` tải `yolov8s-world.pt` để gánh 5 quả còn lại (Chuối, Táo, Cam, Nho, Dâu tây) bằng kỹ thuật mô tả từ khóa (ví dụ: *"small red strawberry with seeds"*).
     - Viết thêm hàm _deduplicate để gộp kết quả của 2 con AI này mà không bị đè viền (bounding box) lên nhau.

2. **`backend/fastapi_app.py` (CHỈNH SỬA CODE)**
   - **Đã thay đổi:** 
     - Sửa `confidence_threshold` mặc định từ `0.5` xuống `0.25` tại hàm API upload để bắt các quả nhỏ tốt hơn.
     - Sửa cách lấy tên Class: dùng `detected_classes = list(set(d.get('class', d.get('class_name', 'unknown'))))` để tương thích với định dạng mới của module `yolo_detector.py` tránh báo lỗi.
     - Cập nhật thông tin Documentation API thành *YOLO-World Fruit Detection API v4.0*.

3. **`backend/classes.json` (GHI ĐÈ FILE)**
   - **Bản cũ:** 6 class lộn xộn (có cherry, plum, tomato).
   - **Đã thay đổi:** Định nghĩa lại đủ 8 class đúng theo chuẩn dự án:
     `{"0": "banana", "1": "apple", "2": "orange", "3": "grape", "4": "watermelon", "5": "strawberry", "6": "mango", "7": "pineapple"}`

4. **`backend/requirements.txt` & `requirements.txt` ở ngoài cùng (THÊM THƯ VIỆN)**
   - **Đã thay đổi:** Thêm dòng `ultralytics>=8.1.0` và `scipy>=1.10.0` vào cuối file để môi trường bên ngoài khi dùng sẽ hiểu cần bộ YOLO-World thần thánh này.

### 📁 Thư mục `frontend/`

5. **`frontend/detect_web.html` (CHỈNH SỬA CODE)**
   - **Đã thay đổi:** 
     - Sửa `value="0.1"` thành `value="0.25"` ở thanh kéo Độ tự tin.
     - Thu gọn mảng Javascript `fruitEmojis` thành đúng 8 trái cây mà AI mới hỗ trợ (xóa cherry, plum, tomato ảo).
     - Thêm một block comment nổi bật dặn bạn cách sửa biến `API_BASE_URL` sau này từ Local URL sang Render URL khi đưa lên máy chủ mạng.

### 📁 Phần Trọng Lượng & Tài Nguyên (AI Models)

6. **Tái cấu trúc thư mục weights**
   - Đã tạo: Khu vực `backend/weights/`
   - Đạt được chuẩn: Có 2 file trọng lượng song hành:
     - `yolov8s-world.pt` (27MB): Cho Nhận diện chung.
     - `best.pt` (64MB): AI Custom cũ của bạn, tôi mới cho lệnh tải lại từ HuggingFace vào thẳng đây chuyên dùng cho mảng "Dưa, Xoài, Dứa".

### 📁 Kiểm Định Bổ Sung (Test Môi Trường Trong)

7. **Các file script dùng để test (MỚI TẠO)**
   - Tạo thư mục giả lập `test_images/` chứa 8 ảnh kiểm thử.
   - Thêm `backend/test_8fruits.py` và `test_real_photos.py` ngoài root để tự động quét 8 ảnh vào Model để ghi biên bản report.

---
**Tóm lại:** Bạn không cần tìm branch của Git nữa. Mọi thứ file tôi nhắc ở trên **đang lập tức nằm ở chính thư mục hiện tại của bạn**. Mở xem trên cửa sổ VS Code lúc này chính là mã code chuẩn xịn cuối cùng đấy!
