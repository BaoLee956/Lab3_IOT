# Mô hình triển khai

`model.onnx` là mô hình YOLO11n đã huấn luyện cho ba trạng thái khẩu trang,
được kèm trong repository để ứng dụng có thể khởi động sau khi clone.

- Kích thước: 10.469.354 byte.
- Đầu vào: FP32 `[1, 3, 416, 416]`.
- Đầu ra: `[1, 7, 3549]`.
- Thứ tự nhãn: `with_mask`, `without_mask`, `mask_weared_incorrect`.
- SHA256: `ea0296a6907d5c260f4c64eeb33e2b4be626bb5775f69253b9228a4a8578ebb3`.

Thông tin xuất mô hình và kết quả đánh giá được lưu trong `evidence/`.
Chạy `train.py` để huấn luyện và xuất lại mô hình khi cần.
