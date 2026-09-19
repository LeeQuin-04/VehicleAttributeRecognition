# 🚗 Vehicle Attribute Recognition

Hệ thống nhận diện phương tiện giao thông theo thời gian thực: **Phát hiện → Theo dõi → Phân loại Loại xe / Màu sắc / Hãng xe**.

Được phát triển trong khuôn khổ **Thực tập tốt nghiệp tại Công ty MQ Solutions** bởi Nhóm 2.

---


## 🏗️ Kiến trúc hệ thống

```
Video Input (MP4 / Camera RTSP)
        │
        ▼
┌───────────────────┐
│ YOLOv8n Detection │  ← Custom-trained trên BDD100K + UA-DETRAC
│(Car / Bus / Truck)│    14.925 train / 9.904 val images
└─────────┬─────────┘
          │  Bounding Boxes
          ▼
┌───────────────────┐
│ ByteTrack Tracker │  ← Kalman Filter + Hungarian Algorithm
│ (Track ID ổn định)│    Xử lý che khuất, duy trì ID
└─────────┬─────────┘
          │  Cropped ROI + Track ID
          ▼
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌────────┐  ┌────────┐
│ Color  │  │  Make  │
│ResNet50│  │ResNet50│
│(15 cls)│  │(VN cars│
└────┬───┘  └────┬───┘
     │           │
     └─────┬─────┘
           ▼
  Temporal Voting (5 frames)
           │
           ▼
  "ID 3  Car  White  Toyota"
           │
           ▼
  Streamlit Web Interface
```

---

## ✨ Tính năng

- **Real-time Detection**: Phát hiện Car / Bus / Truck trên video với YOLOv8n (~65 FPS trên GPU T4)
- **Stable Tracking**: Duy trì Track ID ổn định khi xe bị che khuất (ByteTrack)
- **Color Recognition**: Phân loại 15 màu xe (trắng, đen, đỏ, xanh, bạc, vàng, v.v.)
- **Make Recognition**: Nhận diện hãng xe tối ưu cho xe Việt Nam
- **Temporal Voting**: Kết quả ổn định nhờ majority vote từ 5 frame liên tiếp
- **Web Interface**: Giao diện Streamlit dễ sử dụng, upload video và xem kết quả real-time

---

## 📁 Cấu trúc thư mục

```
VehicleAttributeRecognition/
│
├── app_streamlit.py              # Giao diện web Streamlit
├── requirements.txt              # Thư viện cần cài đặt
├── yolov8n.pt                    # Base weights YOLOv8n
│
├── src/
│   ├── tracking/
│   │   └── video_pipeline.py     # Pipeline chính: YOLO → ByteTrack → Classify
│   │
│   ├── dataset/
│   │   ├── vehicle_dataset.py    # Dataset class
│   │   ├── crop_cars.py          # Script crop xe từ video
│   │   └── yolo_detection/       # Scripts xây dựng YOLO dataset
│   │       ├── prepare_yolo_dataset.py  # Convert BDD100K + UA-DETRAC → YOLO
│   │       ├── pack_for_colab.py        # Đóng gói ZIP upload Colab
│   │       └── create_txt_list.py       # Tạo danh sách train/val
│   │
│   ├── inference/
│   │   └── predictor.py          # Điều phối inference Color + Make
│   │
│   ├── prepare_miotcd.py         # Chuẩn bị dataset MIO-TCD
│   ├── analyze_vehicle_color.py  # Training Color classification
│   ├── analyze_vehicle_make.py   # Training Make classification
│   └── demo_module1.py           # Demo độc lập phần Detection + Tracking
│
├── trained_models/
│   ├── best_yolo_vehicle.pt      # YOLOv8n đã train (Car/Bus/Truck)
│   ├── best_vehicle_type_resnet50.pth
│   ├── best_vehicle_color_resnet50.pth
│   └── best_vehicle_make_resnet50.pth
│
└── data/
    ├── raw/                      # Dataset thô
    ├── unified/                  # Dataset đã xử lý (CSV format)
    └── pseudo_labeled/           # Ảnh gán nhãn bán tự động
```

---

## 🚀 Cài đặt & Chạy

### Yêu cầu hệ thống

- Python 3.9+
- CUDA GPU (khuyến nghị) hoặc CPU

### 1. Clone repo và cài thư viện

```bash
git clone https://github.com/LeeQuin-04/VehicleAttributeRecognition.git
cd VehicleAttributeRecognition
pip install -r requirements.txt
```

### 2. Chạy giao diện Web (Streamlit)

```bash
streamlit run app_streamlit.py
```

Truy cập: `http://localhost:8501`

### 3. Chạy pipeline qua CLI

```bash
python src/tracking/video_pipeline.py --input video_input.mp4 --output video_output.mp4
```

### 4. Demo riêng Detection + Tracking

```bash
# Sửa MODEL_PATH và SOURCE_VIDEO_PATH trong file trước khi chạy
python src/demo_hieu_solo.py
```

---

## 📊 Kết quả mô hình

### YOLOv8n — Vehicle Detection (Car / Bus / Truck)

| Class | Precision | Recall |
|-------|-----------|--------|
| Car   | 0.934     | 0.864  |
| Bus   | 0.887     | 0.851  |
| Truck | 0.908     | 0.864  |

- Tốc độ: ~65 FPS trên GPU T4 / 8–12 FPS trên CPU
- Dataset: 14.925 train + 9.904 val ảnh (BDD100K + UA-DETRAC)

### ResNet50 — Vehicle Color (15 classes)

| Metric | Giá trị |
|--------|---------|
| Test Accuracy | **87.4%** |
| Dataset | 7.267 train / 1.550 val / 1.556 test |

---

## 🗂️ Dataset & Tài nguyên

| Dataset | Dùng cho | Định dạng |
|---------|----------|-----------|
| BDD100K | YOLO Detection | JSON |
| UA-DETRAC | YOLO Detection | XML |
| MIO-TCD | Type Classification | Folder |
| CompCars Surveillance | Make Classification | CSV |

---

## 🛠️ Công nghệ sử dụng

| Lĩnh vực | Thư viện |
|----------|----------|
| Object Detection | Ultralytics YOLOv8 |
| Multi-Object Tracking | Supervision (ByteTrack) |
| Classification | PyTorch · ResNet50 |
| Video Processing | OpenCV |
| Web Interface | Streamlit |
| Training | Google Colab (T4 GPU) |

---

