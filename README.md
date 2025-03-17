# Sign Language Recognition (Nhận dạng ngôn ngữ ký hiệu)

---

# Giới thiệu
Dự án này nhằm xây dựng mô hình nhận dạng ngôn ngữ ký hiệu dựa trên học sâu, sử dụng 3D CNN và YOLOv5 để chuyển đổi ký hiệu thành văn bản trong thời gian thực.

## Công nghệ sử dụng
Python
TensorFlow/Keras (huấn luyện mô hình)
OpenCV (xử lý ảnh/video)
YOLOv5 (phát hiện bàn tay)
3D CNN (phân loại cử chỉ)

## Cấu trúc thư mục
📂 MP_Data/ → Dữ liệu huấn luyện và kiểm thử
📂 pycache/ → Cache của Python
📜 RealTime3D.py → Nhận dạng ký hiệu theo thời gian thực
📜 train_model.py → Huấn luyện mô hình
📜 evaluate_model.py → Đánh giá mô hình
📜 check_data.py → Kiểm tra dữ liệu đầu vào
📜 collect_data.py → Thu thập dữ liệu mới
📜 model3d.h5 → Mô hình đã huấn luyện

## Cách chạy chương trình
### Cài đặt thư viện

pip install -r requirements.txt
### Thu thập dữ liệu

python collect_data.py
### Huấn luyện mô hình

python train_model.py
### Nhận dạng ký hiệu theo thời gian thực

python RealTime3D.py

## Kết quả & Ứng dụng
- Hệ thống có thể nhận diện các ký hiệu ASL với độ chính xác cao.
- Ứng dụng trong giáo dục và giao tiếp với người khiếm thính.

