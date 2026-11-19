# IoT Lab 3 — Nhận dạng trạng thái khẩu trang

Ứng dụng lấy ảnh từ camera trình duyệt, gửi tới FastAPI và chạy mô hình
YOLO11n ONNX trên CPU để phát hiện ba trạng thái:

- `with_mask`: đeo khẩu trang đúng.
- `without_mask`: không đeo khẩu trang.
- `mask_weared_incorrect`: đeo khẩu trang chưa đúng.

Repository kèm mô hình đã huấn luyện tại `assets/model.onnx` (khoảng 10 MB),
giao diện web, mã huấn luyện, kiểm thử và cấu hình triển khai Docker/Render.
Cảnh báo qua Telegram và lịch sử trên Supabase là các tính năng tùy chọn.

## Chạy trên máy

Khuyến nghị Python 3.11. Trong PowerShell, tại thư mục repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000` và cho phép sử dụng camera. Kiểm tra trạng thái
mô hình tại `http://127.0.0.1:8000/health`; tài liệu API tại `/docs`.

`.env.example` liệt kê các biến cấu hình. Ứng dụng đọc biến môi trường của
tiến trình và **không tự nạp file `.env`**. Ví dụ đặt biến trong PowerShell
trước khi chạy Uvicorn:

```powershell
$env:ORT_THREADS = "1"
$env:ALERTS_ENABLED = "0"
$env:MODEL_PATH = "assets/model.onnx"
```

Muốn dùng cảnh báo và lịch sử, làm theo `HUONG_DAN_CANH_BAO.md`, tạo bảng
bằng `supabase/alerts.sql` và đặt thông tin xác thực trong môi trường chạy.
API đọc lịch sử/ảnh yêu cầu cấu hình `API_KEY`.

## Triển khai

`Dockerfile` và `render.yaml` nằm ngay tại gốc repository. Khi kết nối
repository với Render, chọn runtime Docker, nhánh `main`, để trống Root
Directory và dùng `/health` làm đường dẫn kiểm tra sức khỏe.
Xem các bước chi tiết trong `TRIEN_KHAI_RENDER.md`.

```powershell
docker build -t iot-lab3-mask .
docker run --rm -p 8000:8000 -e PORT=8000 iot-lab3-mask
```

Repository được chuẩn bị để triển khai; việc đẩy lên GitHub không tự tạo
dịch vụ Render hay cấu hình Telegram/Supabase.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

Các kiểm thử dùng mô hình ONNX tổng hợp và dịch vụ giả lập để kiểm tra API,
suy luận, cảnh báo, lịch sử và ảnh. Chúng không đo độ chính xác trên camera.

## Huấn luyện và tài liệu

- `prepare_data.py`, `review_data.py`: chuyển nhãn, chia tập và rà soát dữ liệu.
- `train.py`, `requirements-train.txt`, `TRAIN_COLAB.ipynb`: huấn luyện và xuất ONNX.
- `pack_data.py`: đóng gói mã nguồn và dữ liệu đã chuẩn bị để dùng trên Colab.
- `summarize_latency.py`: tổng hợp số đo độ trễ do giao diện xuất ra.
- `HUONG_DAN_LAB3.md`: hướng dẫn bài lab và quy trình dữ liệu/huấn luyện.
- `GIAI_THICH_HE_THONG_VA_HOI_DAP_LAB3.md`: giải thích hệ thống và hỏi đáp.
- `evidence/`: thông tin mô hình, kiểm tra dữ liệu và kết quả đánh giá đã lưu.

Dữ liệu gốc từ [Face Mask Detection trên Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection).
Dataset không được đưa vào Git. Để huấn luyện lại, tải dataset, cài
`requirements-train.txt`, rồi chạy `python prepare_data.py` với đường dẫn
ZIP phù hợp (xem `--help`) và `python train.py`. Notebook Colab sử dụng
các tệp ZIP được tạo sau bước chuẩn bị dữ liệu.

Một số tài liệu chi tiết ghi trạng thái ở thời điểm soạn ban đầu. Mô hình
hiện có được mô tả trong `assets/README.md` và `evidence/model-info.json`;
các số liệu trong `evidence/` là kết quả đã lưu, chưa được đo lại trong
lần chuẩn bị repository này.

Mật khẩu, token, môi trường ảo, cache, dataset, kết quả huấn luyện dung
lượng lớn, bản đóng gói ZIP và bản sao `deploy-render/` được loại khỏi Git.
