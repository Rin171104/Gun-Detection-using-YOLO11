# Gun Detection using YOLO11

## Mô tả
Dự án phát hiện súng (Gun Detection) sử dụng YOLO11. Bao gồm mã huấn luyện, mô hình, và dữ liệu mẫu trong thư mục `gun_dataset/`.

**Repo gốc (đã push):** https://github.com/Rin171104/Gun-Detection-using-YOLO11.git

## Cấu trúc chính
- `train_by_yolo11.py` - script huấn luyện.
- `test_video.py` / `main.py` - script kiểm thử/inference trên video.
- `gun_dataset/` - dữ liệu (images, labels, train/test splits).
- `models/`, `runs/`, `trained-models/` - nơi lưu trọng số và kết quả huấn luyện.

## Yêu cầu
- Python 3.8+
- PyTorch (phiên bản phù hợp với CUDA nếu dùng GPU)
- OpenCV
- NumPy

Ví dụ cài đặt nhanh (tùy môi trường):

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy matplotlib
```

Nếu dự án có `requirements.txt`, dùng:

```bash
pip install -r requirements.txt
```

## Dữ liệu
Thư mục dữ liệu chính là `gun_dataset/`. Thông thường cấu trúc:

- `gun_dataset/train/images` và `gun_dataset/train/labels`
- `gun_dataset/test/images` và `gun_dataset/test/labels`

Lưu ý: thư mục chứa nhiều file nhãn và ảnh → lớn. Khuyến nghị dùng Git LFS cho ảnh và trọng số mô hình.

## Cách sử dụng (ví dụ)
1. Chuẩn bị môi trường và cài phụ thuộc.
2. Kiểm tra/tuỳ chỉnh cấu hình huấn luyện trong `train_by_yolo11.py` hoặc file config tương ứng.
3. Chạy huấn luyện:

```bash
python train_by_yolo11.py
```

4. Kiểm thử/inference trên video hoặc webcam:

```bash
python test_video.py --source path/to/video.mp4
# hoặc
python main.py
```

Tùy script có thể yêu cầu tham số khác; mở file `train_by_yolo11.py` để xem các tùy chọn cụ thể.

## Ghi chú về tệp lớn
- Nếu repo chứa ảnh và mô hình lớn, hãy cân nhắc sử dụng Git LFS cho các file `.jpg`, `.png`, `.pt`, `.weights`.
- Ví dụ cài Git LFS và track:

```bash
git lfs install
git lfs track "*.pt" "*.jpg" "*.png"
git add .gitattributes
```

## Đóng góp
- Fork repo, tạo branch cho tính năng/sửa lỗi, rồi tạo pull request.

## License & Liên hệ
- Kiểm tra file `LICENSE` trong repo (nếu có). Nếu không có, thêm LICENSE phù hợp trước khi công khai.
- Liên hệ: chủ repo trên GitHub: `Rin171104`.
