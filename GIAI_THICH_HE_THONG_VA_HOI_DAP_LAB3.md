# Hiểu hệ thống và chuẩn bị trả lời câu hỏi khi demo Lab 3

Tài liệu dành cho việc demo trực tiếp với giảng viên. Nội dung được đối chiếu với dataset, mã nguồn, cấu hình và kết quả huấn luyện đang có trong dự án. Bạn đã xác nhận hệ thống chạy trên Render và nhận dạng được ba nhãn trong cùng một khung hình. Ba số đo Cloud ở mục 9 là số bạn cung cấp, không phải số giả lập.

## Đọc nhanh trước khi demo

Bổ sung ngày 13/10/2026: mã nguồn có tích hợp cảnh báo Supabase + Telegram tùy chọn; cần tạo dịch vụ và cấu hình trước khi bật. Các số đo ba mẫu bên dưới là của bản nhận dạng trước tích hợp, không phải số đo thời gian gửi Telegram. Xem [hướng dẫn cảnh báo](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/HUONG_DAN_CANH_BAO.md).

- **Bài toán:** phát hiện từng khuôn mặt và xác định trạng thái khẩu trang; một ảnh có thể có nhiều người thuộc những lớp khác nhau.
- **Dataset:** Face Mask Detection trên Kaggle, tài khoản `andrewmvd`; 853 ảnh, 853 file XML và 4.072 khuôn mặt được gán nhãn.
- **Ba lớp:** `with_mask`, `without_mask`, `mask_weared_incorrect`.
- **Mô hình:** YOLO11n, khởi tạo từ `yolo11n.pt` đã được huấn luyện trước rồi tinh chỉnh cho ba lớp khẩu trang.
- **Huấn luyện:** Python + Ultralytics 8.3.230 trên Google Colab, dùng GPU CUDA; hoàn thành 50 epoch, batch 8, kích thước 416.
- **Triển khai:** xuất một file `model.onnx`, chạy ONNX Runtime trên CPU của Render, API FastAPI, giao diện HTML/JavaScript trên laptop.
- **Kết quả ONNX trên test:** mAP@50 = **62,12%**, mAP@50:95 = **37,69%**; lớp đeo sai còn bỏ sót nhiều.
- **Ba mẫu Cloud:** thời gian suy luận trung bình **246,21 ms**, gửi–nhận **515,17 ms**, toàn trình **531,60 ms**.
- **Đánh giá tốc độ:** riêng bước suy luận tương đương khoảng **4,06 ảnh/giây** nếu chạy tuần tự liên tục trong cùng điều kiện; FPS ảnh kết quả thực tế bạn ghi nhận là **1,67–1,79**.
- **Điểm cần nhớ:** thời gian suy luận nằm trong thời gian gửi–nhận, gửi–nhận nằm trong toàn trình. Không cộng cả ba lại và không gọi phần chênh lệch là toàn bộ độ trễ mạng.

## 1. Hệ thống giải quyết bài toán gì?

Hệ thống nhận ảnh từ camera laptop, gửi ảnh tới API trên Cloud, phát hiện khuôn mặt và trả về vị trí cùng trạng thái khẩu trang của từng người. Laptop vẽ các khung và nhãn trên ảnh đã gửi.

**Đây là object detection, tức phát hiện đối tượng.** Đầu ra của mỗi đối tượng gồm bounding box, nhãn và confidence. Hệ thống không chỉ chọn một nhãn duy nhất cho toàn bộ ảnh. Vì vậy trong một khung hình có thể đồng thời có người đeo đúng, người không đeo và người đeo sai.

Một khuôn mặt được chọn một trong ba nhãn. Hệ thống không nhận dạng danh tính hoặc tên của người đó.

**Câu trả lời ngắn:** “Em dùng mô hình phát hiện đối tượng để định vị và phân loại từng khuôn mặt, nên một ảnh có thể trả về nhiều đối tượng với ba trạng thái khẩu trang khác nhau.”

## 2. Dataset lấy từ đâu và chứa những gì?

### 2.1. Nguồn dữ liệu thực sự dùng để train

Dataset là [Face Mask Detection trên Kaggle — andrewmvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection). Đây cũng là dataset được dẫn trong slide lab. Metadata Kaggle đã tải về công bố giấy phép **CC0: Public Domain**.

Dataset đã giải nén và kiểm tra thực tế trong dự án có:

- **853 ảnh PNG:** ảnh đầu vào cho mô hình.
- **853 file XML định dạng Pascal VOC:** mỗi ảnh có một file mô tả kích thước ảnh, tên lớp và tọa độ khung của các khuôn mặt.
- **4.072 đối tượng được gán nhãn:** một ảnh có thể chứa nhiều khuôn mặt, nên số đối tượng lớn hơn số ảnh.

Số đối tượng của từng lớp:

- `with_mask`: **3.232**, khoảng **79,37%**; đeo khẩu trang đúng cách.
- `without_mask`: **717**, khoảng **17,61%**; không đeo khẩu trang.
- `mask_weared_incorrect`: **123**, khoảng **3,02%**; đeo khẩu trang chưa đúng cách, chẳng hạn không che mũi.

Tên `mask_weared_incorrect` được giữ nguyên theo dataset để thứ tự lớp nhất quán từ train đến API.

Các số trên là **số bounding box**, không phải số ảnh riêng của từng lớp. Một ảnh có thể đồng thời chứa nhiều lớp. Nguồn kiểm đếm: [dataset-audit.json](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/evidence/dataset-audit.json).

**Câu trả lời ngắn:** “Em dùng dataset có sẵn trên Kaggle gồm 853 ảnh và nhãn XML cho từng khuôn mặt. Dữ liệu có đủ ba lớp, tổng cộng 4.072 đối tượng; lớp đeo sai ít nhất, chỉ 123 đối tượng.”

### 2.2. Nhãn được chuyển đổi như thế nào?

XML gốc chứa tên lớp và tọa độ pixel `xmin, ymin, xmax, ymax`. Trước khi train, `prepare_data.py` chuyển sang định dạng YOLO: mỗi khuôn mặt là một dòng TXT gồm:

```text
class_id x_center y_center width height
```

Thứ tự lớp cố định: **0 = đeo đúng; 1 = không đeo; 2 = đeo sai**. Tọa độ được chuẩn hóa theo kích thước ảnh:

```text
x_center = (xmin + xmax) / (2 × chiều_rộng_ảnh)
y_center = (ymin + ymax) / (2 × chiều_cao_ảnh)
width    = (xmax - xmin) / chiều_rộng_ảnh
height   = (ymax - ymin) / chiều_cao_ảnh
```

Nhờ chuẩn hóa, nhãn không phụ thuộc trực tiếp vào số pixel của ảnh. File YAML chỉ đường dẫn các tập và ánh xạ ID sang tên lớp. [Định dạng dataset YOLO](https://docs.ultralytics.com/datasets/detect/)

### 2.3. Chia train, validation và test

Tỉ lệ mục tiêu là 70% / 15% / 15%, seed 42. Kết quả thực tế sau khi giữ các nhóm ảnh tương tự cùng tập:

- **Train: 600 ảnh**, gồm 2.242 đối tượng đeo đúng, 503 không đeo, 87 đeo sai. Dùng để cập nhật trọng số.
- **Validation: 131 ảnh**, gồm 466 đeo đúng, 96 không đeo, 19 đeo sai. Dùng theo dõi và chọn checkpoint trong quá trình huấn luyện.
- **Test: 122 ảnh**, gồm 524 đeo đúng, 118 không đeo, 17 đeo sai. Dùng đánh giá mô hình đã chọn.

Tỉ lệ thực tế không chính xác tuyệt đối 70/15/15 vì cần giữ nguyên nhóm ảnh. Tổng vẫn là 853 ảnh. Nguồn: [split-summary.json](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/data/yolo/split-summary.json).

Trước khi chia, dự án dùng SHA256 tìm ảnh trùng file và dHash tìm ảnh có nội dung gần nhau, với ngưỡng khoảng cách 5 bit. Các ảnh liên quan được gom nhóm rồi mới chia. Có 834 nhóm và 19 cặp tương tự theo tiêu chí đó.

Mục đích là giảm **rò rỉ dữ liệu**: nếu ảnh gần như giống nhau xuất hiện cả ở train và test, kết quả test có thể cao vì mô hình đã thấy nội dung tương tự. dHash chỉ là phương pháp gần đúng; nó không chứng minh mọi người hoặc mọi phiên chụp đều đã được tách hoàn toàn.

Dataset bị lệch lớp khá mạnh. Đây là hạn chế thực tế cần nêu khi thầy hỏi tại sao lớp đeo sai yếu. Dự án **không dùng dữ liệu tự thu bằng Teachable Machine** cho lần train này; FMLD chỉ được khảo sát, không dùng huấn luyện.

## 3. Huấn luyện mô hình nào, bằng công cụ nào?

### 3.1. Mô hình YOLO11n

Mô hình là **YOLO11n phiên bản detection**. Chữ `n` là biến thể nano, có quy mô nhỏ trong họ YOLO11, được chọn để giảm nhu cầu tính toán cho Cloud tài nguyên hạn chế. [Nguồn YOLO11](https://docs.ultralytics.com/models/yolo11/)

Khởi tạo bằng `YOLO("yolo11n.pt")`: trọng số ban đầu đã học từ COCO, sau đó được tinh chỉnh trên dataset khẩu trang. Đây là **transfer learning/fine-tuning**, không phải học toàn bộ từ trọng số ngẫu nhiên. Mô hình triển khai cuối cùng dự đoán ba lớp của bài lab, không giữ đầu ra 80 lớp COCO.

Ở mức khái quát, mạng trích xuất đặc trưng ảnh, kết hợp thông tin ở nhiều mức kích thước rồi dự đoán vị trí và điểm số các lớp. Vì xử lý cả ảnh, nó có thể tìm nhiều khuôn mặt trong một lần chạy. Dự án không có bước chạy một mô hình nhận diện khuôn mặt riêng rồi cắt từng mặt để phân loại.

### 3.2. Phân biệt vai trò công cụ

- **Google Colab:** môi trường notebook để chạy mã huấn luyện. Cấu hình lưu lại là `device: '0'`, tức dùng GPU CUDA; không có đủ thông tin để khẳng định loại GPU cụ thể. [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **Python, Ultralytics 8.3.230 và PyTorch:** tạo mạng, train, validation và lưu checkpoint.
- **ONNX:** định dạng mô hình xuất ra để triển khai.
- **ONNX Runtime 1.23.2:** chạy file ONNX trên CPU của Render; không thực hiện huấn luyện. [ONNX Runtime](https://onnxruntime.ai/docs/)
- **FastAPI + Uvicorn:** tiếp nhận yêu cầu HTTP và trả kết quả suy luận.
- **Docker + Render:** đóng gói dependencies, giao diện và mô hình để chạy API trên Cloud.

**Câu trả lời ngắn:** “Em fine-tune YOLO11n bằng Ultralytics trên Colab GPU. Sau đó xuất sang ONNX và dùng ONNX Runtime để suy luận trên CPU Render.”

## 4. Mô tả sơ bộ quá trình huấn luyện

Quy trình thực tế:

1. Chuyển XML sang nhãn YOLO và cố định ba tập dữ liệu.
2. Tải trọng số pretrained `yolo11n.pt`, cấu hình bài toán ba lớp.
3. Đọc các batch ảnh và nhãn từ train; biến đổi ảnh và tọa độ nhãn tương ứng.
4. Chạy mạng để dự đoán bounding box và điểm số lớp.
5. So sánh với nhãn thật để tính loss; lan truyền ngược và cập nhật trọng số.
6. Đánh giá trên validation sau mỗi epoch, giữ checkpoint tốt nhất thành `best.pt`.
7. Đánh giá checkpoint đã chọn trên test, xuất ONNX và đánh giá ONNX trên test.
8. Đưa một file `assets/model.onnx` vào bộ triển khai Render.

**Epoch** là một lượt đi qua tập train. **Batch** là số ảnh xử lý trong một bước cập nhật. Train làm thay đổi trọng số; validation và inference sử dụng trọng số đã có để dự đoán. [Huấn luyện với Ultralytics](https://docs.ultralytics.com/modes/train/)

Cấu hình đã lưu của lần train:

- `epochs = 50`: nhật ký xác nhận đã hoàn thành đủ 50 epoch.
- `batch = 8`: mỗi bước dùng batch tối đa 8 ảnh.
- `imgsz = 416`: kích thước xử lý mục tiêu, đồng thời model ONNX có đầu vào cố định 416 × 416.
- `device = 0`: GPU CUDA trên Colab.
- `seed = 42`, `deterministic = True`: hỗ trợ khả năng lặp lại; không bảo đảm mọi phần cứng cho kết quả tuyệt đối giống nhau.
- `patience = 12`: có thể dừng sớm nếu validation không cải thiện đủ lâu; lần này vẫn chạy hết 50 epoch.
- `optimizer = auto`: để Ultralytics lựa chọn; không tự khẳng định optimizer cụ thể khi chưa có log lựa chọn.

Cấu hình có thay đổi màu/độ sáng, dịch chuyển, thay đổi tỉ lệ, lật ngang và mosaic. Các biến đổi này làm ảnh train đa dạng hơn; chúng không tạo thêm nhiều người hoặc tình huống độc lập như thu thập ảnh thật.

Loss gồm phần sai vị trí khung (`box_loss`), sai lớp (`cls_loss`) và phần học phân bố vị trí (`dfl_loss`). Loss giảm cho thấy mạng đang học tốt hơn theo hàm mục tiêu, nhưng không đủ để kết luận chất lượng trên ảnh mới; phải xem test và từng lớp.

Trong `results.csv`, mAP@50:95 trên validation cao nhất ở **epoch 42**, đạt **55,23%**; mAP@50 ở epoch đó là **79,87%**. Đây là kết quả validation, khác với kết quả test ở mục 6. Phiên bản Ultralytics 8.3.230 dùng mAP@50:95 làm chỉ số fitness của detection để chọn mô hình; checkpoint cuối `last.pt` không nhất thiết tốt hơn `best.pt`. [Mã tính fitness đúng phiên bản](https://github.com/ultralytics/ultralytics/blob/v8.3.230/ultralytics/utils/metrics.py#L866)

Nguồn thực tế: [train.py](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/train.py), [args.yaml](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/runs/mask_train/args.yaml), [results.csv](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/runs/mask_train/results.csv).

## 5. Mô hình trên Cloud và luồng xử lý ảnh

### 5.1. File mô hình triển khai

`model.onnx` là một file chứa đồ thị tính toán và trọng số, dung lượng **10.469.354 byte**, khoảng **9,98 MiB**. File được xuất FP32, batch 1, đầu vào cố định, không lượng tử hóa INT8.

Thông tin đầu vào/đầu ra:

- Đầu vào `[1, 3, 416, 416]`: một ảnh, ba kênh RGB, cao/rộng 416, kiểu số thực FP32.
- Đầu ra `[1, 7, 3549]`: một ảnh, 7 thành phần cho mỗi ứng viên — 4 thành phần mô tả khung và 3 điểm số lớp.
- 3.549 là số ứng viên dự đoán trước lọc, **không phải số người phát hiện được**.

SHA256 nhận diện phiên bản mô hình:

```text
ea0296a6907d5c260f4c64eeb33e2b4be626bb5775f69253b9228a4a8578ebb3
```

Nguồn: [model-info.json](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/evidence/model-info.json). API `/health` trả SHA256 để kiểm tra Cloud đang nạp đúng file; SHA256 không phải chỉ số độ chính xác.

File được COPY vào Docker image và nạp một lần khi API khởi động. Khi Render ngủ hoặc khởi động lại, model được nạp lại từ image. Không huấn luyện lại cho mỗi request, cũng không tải model về laptop để chạy suy luận.

### 5.2. Một ảnh đi qua hệ thống như thế nào?

```text
Camera laptop
  → JavaScript lấy mẫu khung hình
  → mã hóa JPEG và POST /predict qua HTTPS
  → FastAPI đọc ảnh, giải mã JPEG
  → resize giữ tỉ lệ + padding thành 416 × 416, chuẩn hóa RGB
  → ONNX Runtime suy luận trên CPU Render
  → lọc confidence, NMS, đổi tọa độ về ảnh đã gửi
  → trả JSON chứa detections và timing
  → laptop vẽ ảnh đã lấy mẫu, bounding box và nhãn
```

Client giới hạn chiều rộng ảnh gửi tối đa **640 pixel**, giữ tỉ lệ và dùng JPEG quality **0,8**. Server xử lý một ảnh chuẩn hóa 416 × 416 bằng letterbox: resize giữ tỉ lệ rồi thêm viền, tránh kéo méo khuôn mặt. Giá trị RGB được chia 255 về khoảng 0–1.

Ngưỡng confidence hiện tại là **0,40**. Chỉ ứng viên đạt ngưỡng mới được giữ. Confidence là điểm số dự đoán của mô hình; một nhãn hiện 90% không đồng nghĩa hệ thống có accuracy 90%.

NMS loại khung trùng với IoU ngưỡng **0,45**, theo kiểu không phân biệt lớp: khung chồng lấp nhiều và điểm thấp hơn bị loại để giảm việc một khuôn mặt có nhiều nhãn. IoU là diện tích giao chia diện tích hợp của hai khung. Ngưỡng NMS 0,45 khác với ngưỡng matching dùng đánh giá AP@50.

Mỗi kết quả có vị trí khung, ID lớp, tên lớp và confidence. Ảnh đã suy luận giữ đúng khung hình đã gửi; không gắn khung cũ lên video mới đang chuyển động. Video trực tiếp và ảnh kết quả vì vậy có FPS khác nhau.

Nguồn xử lý thực tế: [detector.py](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/detector.py), [app.py](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/app.py), [index.html](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/web/index.html).

## 6. Kết quả huấn luyện và ý nghĩa các chỉ số

### 6.1. Hiểu các chỉ số detection

Một dự đoán đúng cần có đúng lớp và vị trí khung khớp đủ với nhãn thật theo tiêu chí đánh giá. Ký hiệu TP là dự đoán đúng, FP là dự đoán sai/thừa, FN là đối tượng bị bỏ sót.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

**Precision** trả lời: “Trong những đối tượng mô hình báo thuộc lớp này, bao nhiêu là đúng?” **Recall** trả lời: “Trong những đối tượng thực sự thuộc lớp này, mô hình tìm được bao nhiêu?” F1 cân bằng hai chỉ số.

**AP** tổng hợp đường precision–recall của một lớp qua các ngưỡng confidence; **mAP** lấy trung bình AP giữa các lớp. mAP@50 dùng ngưỡng khớp khung IoU 0,50. mAP@50:95 lấy trung bình ở các ngưỡng 0,50 đến 0,95, bước 0,05, nên đòi hỏi vị trí khung chính xác hơn. [Giải thích chỉ số YOLO](https://docs.ultralytics.com/guides/yolo-performance-metrics/)

Đây là detection nên không nên đổi mAP thành “phần trăm ảnh được phân loại đúng”. Một ảnh có nhiều khuôn mặt và có thể vừa có dự đoán đúng, vừa sai hoặc bỏ sót.

### 6.2. Kết quả thật của file ONNX

Trên tập test 122 ảnh, file ONNX có:

- Precision trung bình theo lớp: **81,04%**.
- Recall trung bình theo lớp: **58,29%**.
- **mAP@50: 62,12%**.
- **mAP@50:95: 37,69%**.

Theo từng lớp:

- **Đeo đúng:** precision 94,97%; recall 79,35%; AP@50 86,65%.
- **Không đeo:** precision 65,72%; recall 66,10%; AP@50 66,89%.
- **Đeo sai:** precision 82,43%; recall 29,41%; AP@50 32,83%.

Nguồn: [onnx-test-metrics.json](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/evidence/onnx-test-metrics.json). Đây là đánh giá bằng Ultralytics trên Colab. Phiên bản 8.3.230 lấy precision/recall tại ngưỡng đạt đỉnh đường F1 trung bình đã làm mượt; **không phải đánh giá riêng tại confidence 0,40 của API**. [Mã tính chỉ số đúng phiên bản](https://github.com/ultralytics/ultralytics/blob/v8.3.230/ultralytics/utils/metrics.py#L753), [quy ước validation](https://docs.ultralytics.com/modes/val/)

Checkpoint PyTorch trước khi xuất có mAP@50 **63,91%** và mAP@50:95 **39,12%** trên test. Sau xuất ONNX, hai chỉ số giảm khoảng **1,78** và **1,43 điểm phần trăm**. Vì có sai khác đo được, không nói hai định dạng cho kết quả hoàn toàn giống nhau. Nguồn: [test-metrics.json](D:/STUDY/NAM_4_2026-2027/HK1/IoT/lab3/evidence/test-metrics.json).

### 6.3. Nên nhận xét thế nào?

Lớp đeo đúng tốt nhất. Lớp đeo sai có precision khá nhưng recall thấp: mô hình có thể báo đúng khi phát hiện được, tuy nhiên bỏ sót nhiều đối tượng thật. Lớp này chỉ có 87 đối tượng train và 17 đối tượng test, nên cần thận trọng khi khái quát kết quả.

Việc bạn đã demo thành công ba nhãn trong một khung hình chứng minh chức năng hoạt động trong cảnh đó. Nó không mâu thuẫn với recall thấp trên tập test và không chứng minh mọi cảnh đều nhận đúng.

Có cơ sở nghi ngờ lệch lớp góp phần làm lớp đeo sai yếu, nhưng chưa có thí nghiệm kiểm soát để khẳng định đây là nguyên nhân duy nhất. Hướng cải thiện là bổ sung ảnh đeo sai đa dạng, kiểm tra nhãn và lựa chọn cấu hình bằng validation. Tăng epoch một mình không bảo đảm cải thiện.

## 7. Ba loại độ trễ được đo chính xác ở đâu?

### 7.1. Thời gian suy luận trên Cloud — `inference_ms`

Trong `detector.py`, server ghi thời điểm ngay **trước** và ngay **sau** lệnh `session.run()` của ONNX Runtime:

```python
# Trích từ detector.py
ready = time.perf_counter()
raw = self.session.run(None, {self.input_name: tensor})[0]
inferred = time.perf_counter()
inference_ms = (inferred - ready) * 1000
```

Khoảng này đo thời gian thực tế của lần gọi runtime chạy mô hình, gồm overhead của runtime và ảnh hưởng lịch CPU. Nó không gồm truyền ảnh qua Internet, giải mã JPEG, letterbox trước đó, NMS sau đó hay vẽ giao diện.

`perf_counter()` là bộ đếm thời gian độ phân giải cao thích hợp đo khoảng thời gian; nhân 1.000 để chuyển giây sang ms. [Python time.perf_counter](https://docs.python.org/3/library/time.html#time.perf_counter)

**Vì sao dùng chỉ số này đánh giá tốc độ mô hình?** Nó khoanh vùng bước chạy mạng, giúp giảm ảnh hưởng của mạng và giao diện khi so sánh khả năng suy luận. Tuy nhiên đó vẫn là tốc độ **mô hình trên phần cứng/runtime cụ thể**, không phải tốc độ cố định của YOLO11n ở mọi nơi. Chỉ so sánh công bằng khi cùng phần cứng, tải, batch, kích thước và precision.

### 7.2. Thời gian phản hồi local ↔ Cloud — `rtt_ms`

Client đo từ ngay **trước `fetch()` gửi POST /predict** đến khi **đọc xong toàn bộ body phản hồi bằng `response.text()`**:

```javascript
// Trích phần đo từ web/index.html
tSend = performance.now();
response = await fetch(settings.url + '/predict', {
  method: 'POST', body: form,
  headers: settings.key ? {'X-API-Key': settings.key} : {},
  signal: controller.signal
});
body = await response.text();
tReceive = performance.now();
row.rtt_ms = tReceive - tSend;
```

Đây là thời gian phản hồi vòng yêu cầu ở cấp ứng dụng, gồm upload ảnh, truyền và xử lý HTTP, các thời gian chờ, xử lý server, suy luận, tạo phản hồi JSON và download body. Không chỉ đo ping mạng hoặc dừng đo khi mới nhận header.

JPEG và FormData được chuẩn bị **trước** `tSend`, nên thời gian mã hóa JPEG không nằm trong RTT. Parse JSON, vẽ bounding box và chờ cơ hội vẽ diễn ra **sau** `tReceive`, cũng không nằm trong RTT.

`performance.now()` trả thời gian đơn điệu theo ms trên laptop, thích hợp đo duration hơn lấy giờ lịch. [MDN performance.now](https://developer.mozilla.org/en-US/docs/Web/API/Performance/now)

### 7.3. Độ trễ toàn trình — `e2e_ms`

Client ghi `t0` ngay trước khi sao chép khung hình từ video sang canvas. Sau khi nhận JSON, client vẽ ảnh và bounding box rồi chờ hàm `paintOpportunity()` dùng hai lần `requestAnimationFrame()`:

```javascript
// Các mốc đo thực tế trong web/index.html
const t0 = performance.now();
// Lấy mẫu ảnh → mã hóa JPEG → gửi → nhận → parse JSON → vẽ ảnh và khung.
const painted = await paintOpportunity();
row.e2e_ms = painted - t0;
```

Do đó E2E gồm công việc client trước gửi, toàn bộ RTT và công việc client sau nhận. Đây là chỉ số gần nhất với thời gian người dùng chờ **cho khung hình đã được lấy mẫu đó**.

Hai lần `requestAnimationFrame()` cung cấp cơ hội trình duyệt vẽ giữa các callback. Chúng **không xác nhận chính xác lúc pixel trên màn hình vật lý phát sáng**. Vì vậy giao diện ghi “toàn trình đến cơ hội vẽ”; nên mô tả đây là phép đo xấp xỉ hiển thị. [MDN requestAnimationFrame](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame)

E2E hiện tại không đo độ trễ cảm biến/buffer camera, thời gian chờ đến lần lấy mẫu tiếp theo, mở trang lần đầu hoặc cấp quyền camera. Nếu người dùng vừa đổi trạng thái khẩu trang ngay sau một lần lấy mẫu, họ còn phải chờ đến ảnh mẫu tiếp theo; thời gian quan sát thay đổi có thể lớn hơn E2E của một ảnh.

### 7.4. Vì sao không cần đồng bộ đồng hồ laptop và Cloud?

Server tự lấy hiệu hai mốc `perf_counter()` trên Cloud. Client tự lấy hiệu hai mốc `performance.now()` trên laptop. Chỉ gửi **duration** từ server về để hiển thị, không lấy timestamp Cloud trừ timestamp laptop.

Thông thường trên cùng request thành công:

```text
T_suy_luận ≤ T_gửi_nhận ≤ T_toàn_trình
```

Các khoảng này bao hàm nhau. Cộng cả ba sẽ tính lặp bước suy luận và các phần của RTT.

### 7.5. Những thời gian server phụ có ý nghĩa gì?

- **`decode_ms`:** giải mã ảnh đã nhận và chuyển sang RGB.
- **`queue_ms`:** chờ khóa suy luận trong hàm xử lý; không đo toàn bộ thời gian chờ ở proxy hoặc thread pool.
- **`preprocess_ms`:** resize/letterbox, tạo tensor và chuẩn hóa.
- **`postprocess_ms`:** giải mã đầu ra, lọc confidence, NMS và đổi tọa độ.
- **`cloud_pipeline_ms`:** từ lúc hàm `infer()` bắt đầu xử lý bytes ảnh đến khi hoàn tất dự đoán; bao gồm các mục trên và suy luận, kèm overhead nhỏ.

Pipeline bắt đầu sau khi API đã đọc dữ liệu ảnh. Nó không gồm toàn bộ đọc multipart, chờ trước khi vào hàm, tạo JSON phản hồi và truyền JSON về laptop. Các số này giúp xác định chậm ở bước nào nếu cần phân tích sâu hơn.

## 8. Từ thời gian suy luận, tính và đánh giá tốc độ thế nào?

### 8.1. Đổi ms/ảnh sang ảnh/giây

Với batch 1 chạy tuần tự liên tục, chỉ xét bước suy luận:

```text
Tốc độ suy luận tương đương ≈ 1000 / T_suy_luận_ms
```

Lý do: một giây có 1.000 ms; nếu mỗi ảnh cần T ms thì số lần suy luận tuần tự trong một giây xấp xỉ 1.000/T. Với nhiều ảnh, tốc độ tương đương của cả nhóm là `N × 1000 / tổng thời gian suy luận`, tức `1000 / thời gian suy luận trung bình`.

Đây là ước lượng của riêng bước suy luận trong điều kiện đã đo, **không phải throughput Cloud đo bằng bài kiểm thử tải**, không phải FPS camera và không phải FPS giao diện. Không tính trung bình các nghịch đảo rồi gọi đó là throughput của cả nhóm.

### 8.2. FPS ảnh kết quả thực tế

Giao diện lưu thời điểm vẽ tối đa 10 ảnh kết quả gần nhất rồi tính:

```text
FPS_kết_quả = 1000 × (số_mốc - 1) / (mốc_cuối - mốc_đầu)
```

Đây là số ảnh có kết quả cập nhật mỗi giây trên một cửa sổ gần đây. Nó phản ánh mạng, toàn bộ xử lý và lịch lấy mẫu; vì vậy khác với `1000/inference_ms` và không nhất thiết bằng `1000/e2e_ms` của một lượt.

Client hiện giới hạn tối đa 2 ảnh/giây. Khoảng lấy mẫu mục tiêu tối thiểu là 500 ms; nếu pipeline chậm, tăng khoảng này theo khoảng hai lần thời gian pipeline quan sát. Mỗi lúc chỉ có một request đang chờ; không xếp hàng hàng loạt ảnh cũ. Khoảng chờ lấy mẫu không nằm trong E2E của ảnh kế tiếp nhưng ảnh hưởng FPS cập nhật.

Giới hạn 2 ảnh/giây nhằm cân bằng cập nhật và tải Cloud. Nó không ngăn mọi tình huống quá tải nếu có nhiều người dùng. Render Free hiện có **0,1 CPU và 512 MB RAM**, nên tốc độ trên GPU Colab không thể dùng thay cho tốc độ Cloud. [Cấu hình Render](https://render.com/docs/compute-plans)

## 9. Phân tích ba số đo Cloud bạn cung cấp

### 9.1. Dữ liệu gốc

Cả ba mẫu đều được giao diện đánh dấu `steady`:

- **Lượt 43:** FPS 1,79; suy luận 210,76 ms; gửi–nhận 483,10 ms; E2E 501,10 ms; JPEG 31.552 byte.
- **Lượt 37:** FPS 1,67; suy luận 229,60 ms; gửi–nhận 505,30 ms; E2E 520,90 ms; JPEG 31.990 byte.
- **Lượt 57:** FPS 1,71; suy luận 298,27 ms; gửi–nhận 557,10 ms; E2E 572,80 ms; JPEG 32.886 byte.

Không có thông tin đủ để kết luận ba mẫu thuộc cùng cảnh hoặc cùng phiên. Đây là ba quan sát bạn gửi để giải thích phép tính, không phải thống kê toàn bộ phiên và không đại diện chắc chắn cho mọi điều kiện mạng.

### 9.2. Trung bình và khoảng quan sát

```text
Suy luận trung bình = (210,76 + 229,60 + 298,27) / 3
                   = 246,21 ms/ảnh

RTT trung bình     = (483,10 + 505,30 + 557,10) / 3
                   ≈ 515,17 ms

E2E trung bình     = (501,10 + 520,90 + 572,80) / 3
                   = 531,60 ms
```

- Suy luận quan sát: **210,76–298,27 ms**, median **229,60 ms**.
- RTT quan sát: **483,10–557,10 ms**, median **505,30 ms**.
- E2E quan sát: **501,10–572,80 ms**, median **520,90 ms**.
- Ba giá trị FPS hiển thị: **1,67–1,79**; trung bình mô tả khoảng **1,72**. Mỗi giá trị là FPS theo cửa sổ gần đây, nên trung bình này không thay thế FPS tính từ mọi mốc của cả phiên.

Mean là trung bình cộng; median là giá trị ở giữa khi sắp xếp. Nếu sau này có nhiều lượt, p95 là phân vị mà khoảng 95% số đo không vượt quá. Ba mẫu hiện tại quá ít để dùng p95 làm kết luận độ ổn định; không cần thêm dữ liệu để thực hiện các phép tính minh họa ở đây.

### 9.3. Đánh giá riêng bước suy luận

```text
Tốc độ tương đương của bước suy luận = 1000 / 246,21
                                    ≈ 4,06 ảnh/giây
```

Theo từng lượt, các giá trị tương đương lần lượt khoảng 4,74; 4,36; 3,35 ảnh/giây. Trong ba quan sát, riêng bước chạy model nhanh hơn chu kỳ 500 ms của mục tiêu 2 ảnh/giây. Tuy nhiên còn tiền/hậu xử lý và các phần của RTT, nên điều đó không bảo đảm giao diện đạt đúng 2 FPS.

Nhận xét hợp lý: “Với Render Free và cấu hình hiện tại, bước suy luận có độ trễ trung bình khoảng 246 ms trong ba mẫu, phù hợp mục tiêu cập nhật chậm khoảng 2 lần/giây của demo. Đây chưa phải tốc độ đủ cho nhận dạng mỗi khung hình video 30 hay 60 FPS.”

Để xử lý riêng bước suy luận mỗi khung hình ở 30 FPS cần khoảng không quá 33,33 ms/ảnh; 60 FPS cần khoảng 16,67 ms/ảnh. Các phép chia này chỉ là ngân sách thời gian, chưa gồm hệ thống xung quanh.

### 9.4. Phần thời gian ngoài suy luận trong RTT

```text
Lượt 43: 483,10 - 210,76 = 272,34 ms
Lượt 37: 505,30 - 229,60 = 275,70 ms
Lượt 57: 557,10 - 298,27 = 258,83 ms

Trung bình RTT - suy luận ≈ 268,96 ms
```

Khoảng này gồm truyền đi/về **và** xử lý HTTP, đọc/giải mã ảnh, tiền/hậu xử lý, hàng đợi, tạo phản hồi và overhead khác. Vì chưa có `cloud_pipeline_ms` của ba mẫu, chưa thể tách riêng công việc server ngoài suy luận.

Không được nói “mạng mất 269 ms” hoặc lấy khoảng này chia hai để suy ra độ trễ một chiều. Ngay cả `RTT - cloud_pipeline_ms` vẫn chứa các phần ngoài pipeline, không phải độ trễ mạng thuần.

### 9.5. Phần thời gian client ngoài RTT

```text
Lượt 43: 501,10 - 483,10 = 18,00 ms
Lượt 37: 520,90 - 505,30 = 15,60 ms
Lượt 57: 572,80 - 557,10 = 15,70 ms

Trung bình E2E - RTT ≈ 16,43 ms
```

Đó là tổng công việc client trước gửi và sau nhận: lấy mẫu, mã hóa JPEG, chuẩn bị dữ liệu, parse JSON, vẽ khung và chờ cơ hội vẽ. Nó không phải chỉ riêng thời gian vẽ.

Có thể diễn giải trung bình toàn trình thành ba phần **không trùng nhau**:

```text
531,60 ms ≈ 246,21 ms + 268,96 ms + 16,43 ms
            suy luận   phần RTT    phần client
                       còn lại    ngoài RTT
```

Riêng suy luận chiếm khoảng **46,3%** E2E trung bình. Phần còn lại hơn một nửa, nên tăng tốc model sẽ có ích nhưng không loại bỏ được mọi độ trễ. Ba mẫu chưa đủ để khẳng định mạng là nút thắt lớn nhất.

### 9.6. Đánh giá trải nghiệm người dùng và kích thước ảnh

Người dùng chờ khoảng **nửa giây** cho kết quả của một ảnh đã lấy mẫu. Với bài demo trạng thái khẩu trang tương đối ít thay đổi, mức này có thể chấp nhận được; với chuyển động nhanh, kết quả sẽ có độ trễ dễ nhận thấy. Đánh giá “phù hợp” ở đây gắn với mục tiêu demo, không phải một tiêu chuẩn phổ quát.

FPS 1,67–1,79 tương ứng khoảng 0,56–0,60 giây giữa các lần cập nhật trên cửa sổ đo, khá gần mục tiêu tối đa 2 FPS. FPS thực thấp hơn mức suy luận tương đương 4,06 vì hệ thống còn truyền dữ liệu, xử lý ngoài model và chủ động điều tiết lấy mẫu.

JPEG khoảng 31–33 nghìn byte trong các mẫu, giúp giảm lượng upload so với gửi ảnh không nén. Số byte là dung lượng ảnh, không phải tốc độ mạng; quality 0,8 cũng không phải confidence 80%. Nén quá mạnh có thể làm mất chi tiết khẩu trang.

**Câu nhận xét có thể nói với thầy:** “Trong ba mẫu steady, suy luận trung bình 246,21 ms, phản hồi gửi–nhận 515,17 ms và toàn trình 531,60 ms. Riêng suy luận tương đương khoảng 4,06 ảnh/giây, còn ảnh kết quả cập nhật khoảng 1,67–1,79 FPS do có truyền dữ liệu, xử lý thêm và giới hạn lấy mẫu. Với demo khẩu trang, người dùng chờ khoảng nửa giây cho mỗi ảnh; em chưa coi ba mẫu này là kết luận thống kê toàn bộ hệ thống.”

## 10. Nếu thầy hỏi cách đọc các chỉ số khi đang demo

### “Suy luận tăng nhưng RTT không tăng tương ứng thì sao?”

Các phần khác có thể giảm, chẳng hạn chờ ít hơn hoặc truyền nhanh hơn. RTT bao gồm nhiều thành phần; không suy ra một phần thay đổi hoàn toàn từ phần tổng.

### “Inference nhỏ nhưng RTT lớn thì chậm ở đâu?”

Có nhiều thời gian ngoài chạy model. Xem thêm pipeline, decode, queue, tiền/hậu xử lý và tình trạng kết nối. Chưa đo đủ thì chỉ nói chậm ngoài suy luận, không kết luận ngay là mạng.

### “RTT nhỏ nhưng E2E lớn thì sao?”

Xem phía client: mã hóa ảnh, xử lý JSON, vẽ và lịch trình trình duyệt. Tab ẩn hoặc laptop đang bận cũng có thể làm thời điểm vẽ chậm.

### “Vì sao có lúc chờ rất lâu ở lần đầu?”

Render Free ngủ sau 15 phút không có lưu lượng; khởi động lại thường khoảng một phút. Đây là cold start, khác với các lượt steady. Nếu mở giao diện Cloud trước, thao tác đó đã có thể đánh thức service. Không gộp thời gian mở trang hoặc nạp model vào `inference_ms`. [Giới hạn Render Free](https://render.com/docs/free)

### “Có phải Cloud lấy camera của em không?”

Camera thuộc laptop và được trình duyệt truy cập khi người dùng cho phép. Cloud nhận ảnh mẫu gửi qua HTTPS và trả kết quả. Khi bật tích hợp cảnh báo đã cấu hình, ảnh kích hoạt cảnh báo được lưu trong bucket private trên Supabase và gửi trực tiếp qua Telegram. Khi cảnh báo tắt, đường nhận dạng không lưu ảnh lâu dài.

## 11. Những câu hỏi dễ gặp về lựa chọn và hạn chế

### “Tại sao không dùng Teachable Machine?”

Em cần nhiều khuôn mặt và nhiều nhãn trong một ảnh. Bộ phân loại ảnh thông thường chỉ trả xác suất các lớp cho toàn ảnh, không tự tạo bounding box cho từng người. YOLO detection đáp ứng trực tiếp nhu cầu này.

### “Tại sao không chạy model ngay trên laptop?”

Chạy local có thể giảm độ trễ truyền dữ liệu, nhưng bài này triển khai suy luận qua API Cloud và đo các độ trễ liên quan. Laptop làm phần thu ảnh/hiển thị, Render chạy model. Không khẳng định Cloud luôn nhanh hơn local.

### “Tại sao chọn 416 thay vì 640?”

416 giảm khối lượng tính toán và phù hợp tài nguyên giới hạn. Đổi lại, khuôn mặt nhỏ hoặc xa có thể khó nhận dạng hơn. Tăng kích thước cần đánh giá lại chất lượng và thời gian; không phải lúc nào cũng tốt hơn tổng thể.

### “Tại sao chọn ONNX?”

ONNX cho phép dùng runtime suy luận riêng, không cần cài toàn bộ Ultralytics/PyTorch trên server này. File xuất chứa cả trọng số trong một file, đáp ứng yêu cầu lưu một file mô hình trên Cloud. Đổi định dạng không tự bảo đảm nhanh hơn hoặc giữ nguyên mọi chỉ số; dự án đã có đánh giá ONNX riêng.

### “Tại sao giới hạn một request và một worker?”

Model được nạp một lần vào một worker. Client chỉ gửi ảnh tiếp khi ảnh trước hoàn tất, giảm ảnh cũ chờ xử lý và giảm tải. Khóa suy luận ở server tránh nhiều request cùng chạy model; `ORT_THREADS=1` giới hạn luồng tính toán trong ONNX Runtime. Hai cơ chế này khác nhau. Cấu hình nhằm phù hợp tài nguyên nhỏ, không phải giải pháp phục vụ đông người.

### “Demo ba nhãn tốt rồi thì tại sao kết quả test chưa cao?”

Một cảnh demo có thể sáng, mặt gần và ít che khuất nên dễ hơn nhiều ảnh test. Test đánh giá cả dự đoán sai, bỏ sót và độ khớp bounding box. Nhận đúng một cảnh không có nghĩa recall hay mAP của mọi cảnh đều cao.

### “Muốn cải thiện lớp đeo sai, em làm gì?”

Ưu tiên thêm và kiểm tra nhãn ảnh đeo sai, đa dạng người/ánh sáng/góc/khoảng cách, giữ các phiên tương tự cùng tập. Sau đó lựa chọn ngưỡng hoặc cấu hình bằng validation, đánh giá lại mô hình cuối; không chọn tham số dựa trên test rồi báo test như đánh giá độc lập.

### “Camera laptop có đáp ứng phần IoT không?”

Trong hệ thống này camera là nguồn dữ liệu tại thiết bị đầu cuối, kết nối Internet tới Cloud qua API. Slide có nêu camera thường hoặc camera IoT, đồng thời một phần hỗ trợ nhắc Arduino. Em dùng camera thường của laptop; nếu giảng viên yêu cầu riêng Arduino thì cần xác nhận phạm vi đó. Không nói đã dùng Arduino khi hệ thống không có.

## 12. Bài trình bày khoảng một phút trước khi thao tác demo

“Em xây dựng hệ thống phát hiện trạng thái khẩu trang cho từng khuôn mặt. Dataset lấy từ Kaggle có 853 ảnh, nhãn bounding box XML và ba lớp: đeo đúng, không đeo, đeo sai. Em chuyển nhãn sang YOLO, chia thành 600 ảnh train, 131 validation và 122 test, đồng thời gom ảnh gần giống trước khi chia để giảm rò rỉ dữ liệu.

Em fine-tune YOLO11n bằng Ultralytics trên Colab GPU, chạy 50 epoch với ảnh kích thước 416, batch 8. Sau đó xuất một file ONNX và triển khai FastAPI trên Render CPU. Mô hình ONNX có mAP@50 khoảng 62,12% trên test; lớp đeo sai còn hạn chế do recall thấp, dù hệ thống đã nhận được ba nhãn đồng thời trong cảnh demo.

Camera và giao diện chạy trên laptop, gửi ảnh JPEG lên Cloud, Cloud trả bounding box, nhãn và các thời gian xử lý. Em đo riêng thời gian suy luận trên server, thời gian từ gửi đến đọc xong phản hồi trên client và toàn trình từ lấy mẫu đến cơ hội vẽ. Trong ba mẫu đã quan sát, ba giá trị trung bình khoảng 246, 515 và 532 ms. Giao diện cập nhật khoảng 1,7 ảnh kết quả mỗi giây, phù hợp mục tiêu demo với tài nguyên Cloud giới hạn.”

Khi thầy hỏi số cụ thể, xem mục 9. Khi thầy hỏi chất lượng nhận dạng, xem mục 6. Các nhận xét về hạn chế dựa trên số liệu thật; không thay số confidence của một khung bằng accuracy của toàn mô hình.
