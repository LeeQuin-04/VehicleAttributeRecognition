import cv2
import os
from ultralytics import YOLO

video_path = r"D:\TTTN_Co_Phuong\videoVN.mp4"
output_dir = r"D:\TTTN_Co_Phuong\Dataset\Make_Finetune_Data\raw_crops"
yolo_path = r"D:\TTTN_Co_Phuong\VehicleAttributeRecognition\VehicleAttributeRecognition\trained_models\best_yolo_vehicle.pt"

os.makedirs(output_dir, exist_ok=True)

print("Đang tải YOLO...")
model = YOLO(yolo_path)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Lỗi: Không thể mở video {video_path}")
    exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0: fps = 30

frame_idx = 0
car_count = 0

print("Bắt đầu cắt ảnh xe (cứ mỗi 1 giây cắt 1 lần)...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Lấy mẫu mỗi 1 giây
    if frame_idx % fps == 0:
        # Chỉ detect class 0 (Car) theo model mới
        results = model(frame, classes=[0], verbose=False)[0] 
        
        boxes = results.boxes.xyxy.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Chỉ lấy các xe đủ lớn (tránh nhiễu)
            if (x2 - x1) > 40 and (y2 - y1) > 40:
                crop = frame[y1:y2, x1:x2]
                
                # Lưu file
                out_path = os.path.join(output_dir, f"car_{frame_idx}_{i}.jpg")
                cv2.imwrite(out_path, crop)
                car_count += 1
                
    frame_idx += 1

cap.release()
print(f"XONG! Đã cắt được {car_count} ảnh xe ô tô. Mời bạn vào thư mục: {output_dir}")
