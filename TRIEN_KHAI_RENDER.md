# Bước tiếp theo: đưa API và mô hình đã huấn luyện lên Render

## 1. Chuẩn bị bộ file

Mục tiêu: có đủ API, giao diện hiện tại và model.onnx đã huấn luyện.

Bộ file đã chuẩn bị nằm trong `deploy-render/`; bản nén là `deploy-render.zip`.
Khi thay mô hình hoặc sửa giao diện/API, chạy lại từ thư mục lab3:

```powershell
.\.venv\Scripts\python.exe prepare_render.py
```

Kết quả mong đợi: thông báo `archive_verified: true`. Mã SHA256 của mô hình
nằm trong `deploy-render/deployment-manifest.json`. Không thay tên model.onnx.

## 2. Tải lên GitHub

Mục tiêu: tạo kho mã nguồn để Render lấy bộ triển khai.

1. Đăng nhập https://github.com và tạo repository tên `iot-lab3-mask`.
2. Có thể chọn Private và cấp cho Render quyền truy cập riêng repository này.
3. Trong repository mới, chọn liên kết upload existing files; nếu đã có file,
   chọn Add file > Upload files.
4. Mở thư mục deploy-render trên laptop, chọn toàn bộ NỘI DUNG bên trong và
   kéo vào trang tải lên. Giữ nguyên hai thư mục assets và web.
5. Commit với nội dung `Deploy Lab 3 API and trained ONNX model`.

Kiểm tra thành công: ở gốc repository thấy Dockerfile, app.py, detector.py,
requirements.txt, render.yaml, .dockerignore; trong assets có model.onnx và
trong web có index.html, favicon.svg. Nếu thấy thư mục deploy-render ở gốc
và Dockerfile nằm bên trong, bạn đã tải thừa một cấp thư mục.

GitHub giới hạn tải bằng trình duyệt 25 MiB mỗi file; mô hình hiện tại dưới
giới hạn này. Nguồn: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

## 3. Tạo Web Service trên Render

Mục tiêu: chạy suy luận trên Cloud và mở giao diện bằng HTTPS.

1. Đăng nhập https://dashboard.render.com, chọn New > Web Service.
2. Kết nối GitHub, cấp quyền repository vừa tạo, rồi chọn repository đó.
3. Điền Name: iot-lab3-mask; Branch: main (hoặc nhánh bạn vừa tải lên).
4. Chọn Language/Runtime: Docker. Root Directory để trống.
5. Dockerfile Path: ./Dockerfile. Docker Command để trống để dùng CMD có sẵn.
6. Chọn Region: Singapore nếu có; nếu không, chọn khu vực gần Việt Nam nhất.
7. Chọn Compute Plan/Instance Type: Free.
8. Thêm Environment Variables: ORT_THREADS = 1; MODEL_PATH = assets/model.onnx.
9. Đặt Health Check Path = /health trong Advanced nếu có. Nếu không thấy,
   chỉnh trong Settings sau khi tạo service.
10. Nhấn Deploy Web Service và theo dõi Logs.

Lưu ý: khi tạo Web Service bằng giao diện, hãy tự điền các cấu hình trên;
render.yaml được dùng tự động khi tạo bằng Blueprint, không thay thế các
trường cấu hình của luồng tạo thủ công này.

Không cần Build Command hay Start Command cho phương án Docker. Dockerfile
đã cài dependencies và chạy 1 worker, 1 luồng ONNX, nghe cổng PORT của Render.
Không cần MODEL_URL: model.onnx được COPY vào Docker image, lưu trên Cloud
cùng ứng dụng và nạp một lần khi API khởi động.

Nguồn: https://render.com/docs/docker và https://render.com/docs/web-services

Giới hạn hiện tại: Free có 0,1 CPU, 512 MB RAM; ngủ sau 15 phút không có lưu
lượng và thường cần khoảng 1 phút để khởi động lại. Workspace có tổng 750
giờ Free mỗi tháng; hạn mức bandwidth/build xem ở Billing > Monthly Included
Usage. Có phương thức thanh toán thì vượt một số hạn mức có thể bị tính phí.
Nguồn: https://render.com/docs/compute-plans và https://render.com/docs/free

## 4. Kiểm tra API và camera

Mục tiêu: xác nhận đúng mô hình và ảnh thật được suy luận trên Cloud.

1. Khi service báo Live, mở URL Render cấp, thêm /health ở cuối.
2. Kiểm tra status = ok, đủ ba classes, input_size = 416, provider =
   CPUExecutionProvider. So sánh model_sha256 với deployment-manifest.json.
3. Mở lại URL gốc bằng Chrome/Edge trên laptop. Ô Địa chỉ API phải là chính
   URL Render, không phải 127.0.0.1:8000. Không thêm /predict vào ô này.
4. Đặt Số lượt chính thức = 20 cho lần thử đầu; nhấn Bật camera và đo, cho
   phép camera. Giao diện gửi thêm 5 lượt làm nóng.
5. Thử lần lượt đeo đúng, không đeo và đeo sai. Kiểm tra có kết quả, số đo
   cập nhật, rồi tải CSV và lưu ảnh kết quả.

Camera vẫn là camera laptop; chỉ ảnh mẫu được gửi đi, mô hình chạy trên Render.
Mức tối đa 2 ảnh/giây là mục tiêu của client; tốc độ thực tế có thể thấp hơn.
Client giảm tốc độ theo thời gian xử lý, chỉ có một yêu cầu đang chờ. Cơ chế
này giảm tải nhưng không đảm bảo hạn mức tài nguyên cho nhiều người cùng dùng.

Chưa cần sửa CORS khi mở giao diện tại cùng URL Render. Nếu dùng giao diện
local trên cổng 5500, các origin localhost:5500 đã được cho phép sẵn.

## 5. Nếu có lỗi

- Missing model: kiểm tra assets/model.onnx đã được commit lên GitHub.
- Dockerfile not found: kiểm tra Dockerfile nằm ở gốc; Root Directory trống.
- Build failed: lấy lỗi đầu tiên có nội dung cụ thể trong Build Logs.
- Out of memory: kiểm tra ORT_THREADS=1 và không chạy thêm worker. Nếu vẫn
  lỗi, cần giảm mô hình/tài nguyên sau khi xem log; chưa kết luận Free phù hợp.
- Trang chờ khởi động: chờ service thức dậy rồi mở /health; không tính lần
  thức dậy vào phép đo ổn định, ghi riêng nếu muốn báo cáo cold start.
- Camera không mở: dùng HTTPS Render trên laptop và cho phép camera ở
  thanh địa chỉ; đóng ứng dụng khác đang chiếm camera.
- Lỗi 401: nếu đã đặt API_KEY trên Render, nhập cùng key trong giao diện.
- Lỗi 503: dừng đo, xem trạng thái/log service rồi thử lại; không tăng FPS.

## 6. Minh chứng cần lưu

Ảnh service Live, trang /health (đúng SHA256), ảnh nhận dạng từng lớp và CSV
20 lượt thử. Sau khi ổn định, đo ít nhất 100 lượt chính thức cho từng cảnh,
lưu kết quả 5 lượt warmup riêng và tính thống kê bằng summarize_latency.py.

Chưa có số liệu Cloud cho đến khi bạn triển khai và chạy thực tế.
