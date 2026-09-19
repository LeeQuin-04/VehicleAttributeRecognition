# YOLO Detection Dataset Preparation

Scripts do **Bùi Văn Hiếu** viết để xây dựng bộ dữ liệu cho module YOLOv8 Detection.

## Cấu trúc

```
src/dataset/yolo_detection/
├── prepare_yolo_dataset.py   # Convert BDD100K (JSON) + UA-DETRAC (XML) → YOLO .txt format
├── pack_for_colab.py         # Đóng gói ảnh + label thành YOLO_Dataset_Colab.zip để upload Colab
└── create_txt_list.py        # Tạo danh sách đường dẫn ảnh train/val
```

## Nguồn dữ liệu

| Dataset    | Định dạng annotation | Số lượng ảnh (sau xử lý) |
|------------|----------------------|--------------------------|
| BDD100K    | JSON                 | ~14k frames              |
| UA-DETRAC  | XML (cấu trúc cây)   | ~10k frames (1/10 sample)|

**Dataset YOLO cuối cùng:**
- Train: **14.925 ảnh**
- Val: **9.904 ảnh**
- Background: **0 ảnh**
- Classes: `0:car  1:bus  2:truck  3:motorbike  4:bicycle`

## Cách chạy

### Bước 1: Convert annotation sang YOLO format
```bash
# Chỉnh BASE_DIR, BDD_IMG_DIR, DETRAC_IMG_DIR trong file trước khi chạy
python src/dataset/yolo_detection/prepare_yolo_dataset.py
```

### Bước 2: Tạo danh sách ảnh train/val
```bash
python src/dataset/yolo_detection/create_txt_list.py
```

### Bước 3: Đóng gói để upload Colab
```bash
python src/dataset/yolo_detection/pack_for_colab.py
# Output: YOLO_Dataset_Colab.zip
```

### Bước 4: Train YOLOv8 trên Colab
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='/content/YOLO_Dataset/dataset.yaml', epochs=50, imgsz=640)
```

## Lưu ý
- File `YOLO_Dataset_Colab.zip` (~vài GB) **không được commit** lên git (đã có trong `.gitignore`).
- Weights đã train: `trained_models/best_yolo_vehicle.pt`
