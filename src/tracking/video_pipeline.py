import cv2
import numpy as np
import torch
from pathlib import Path
import supervision as sv
from ultralytics import YOLO

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import Predictor từ file inference đã viết
from inference.predictor import VehiclePredictor

class TrackingPipeline:
    def __init__(
        self,
        video_path: str,
        output_path: str,
        multihead_path: str = None,
        make_model_path: str = None
    ):
        self.video_path = video_path
        self.output_path = output_path
        
        # 1. Khởi tạo YOLOv8 (Detect)
        # Dùng bản nano (yolov8n.pt) cho nhẹ và nhanh. Tự tải về nếu chưa có.
        print("[1/3] Đang tải YOLOv8 (Detect)...")
        self.yolo_model = YOLO("yolov8n.pt")
        # COCO classes: 2=car, 5=bus, 7=truck
        self.target_classes = [2, 5, 7] 
        
        # 2. Khởi tạo ByteTrack (Track) qua supervision
        print("[2/3] Đang khởi tạo ByteTrack (Tracker)...")
        self.tracker = sv.ByteTrack()
        
        # Các công cụ vẽ Box và Text của Supervision
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=2, 
            text_scale=0.5,
            text_position=sv.Position.TOP_CENTER
        )
        
        # 3. Khởi tạo Hybrid Predictor (Classify)
        print("[3/3] Đang tải Hybrid Predictor (Classify)...")
        # Khởi tạo mà không báo lỗi nếu file chưa tồn tại ngay (để chuẩn bị trước)
        try:
            self.predictor = VehiclePredictor(
                multihead_path=multihead_path,
                make_model_path=make_model_path,
                make_threshold=0.4
            )
            self.predictor_ready = True
        except FileNotFoundError as e:
            print(f"[CẢNH BÁO] Không tìm thấy model classify: {e}")
            print("Video pipeline sẽ chỉ chạy Detection + Tracking.")
            self.predictor_ready = False
        # Hệ thống Voting: {track_id: {"type": [], "color": [], "make": []}}
        # Lưu lại lịch sử 5 lần dự đoán gần nhất của từng xe
        self.attr_history = {}
        # Cache chuỗi hiển thị đã được lấy phiếu đa số
        self.attr_display = {}

    def process_frame(self, frame: np.ndarray, frame_index) -> np.ndarray:
        """Xử lý từng frame của video"""
        from collections import Counter
        
        # 1. Phát hiện bằng YOLO
        results = self.yolo_model(frame, classes=self.target_classes, verbose=False)[0]
        
        # Chuyển đổi format YOLO -> Supervision
        detections = sv.Detections.from_ultralytics(results)
        
        # 2. Theo dõi bằng ByteTrack
        detections = self.tracker.update_with_detections(detections)
        
        labels = []
        
        # 3. Phân loại thuộc tính cho từng xe
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            track_id = detections.tracker_id[i]
            
            if track_id not in self.attr_history:
                self.attr_history[track_id] = {"type": [], "color": [], "make": []}
            
            # Cứ mỗi 3 frame, dự đoán lại 1 lần để cập nhật khi xe tiến gần hơn
            # (Giúp model sửa sai nếu lúc xe ở xa bị mờ đoán sai)
            if self.predictor_ready and (frame_index % 3 == 0 or not self.attr_history[track_id]["type"]):
                x1, y1, x2, y2 = map(int, bbox)
                
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Chỉ đoán khi box đủ lớn (> 40x40) để tránh nhiễu
                if (x2 - x1) > 40 and (y2 - y1) > 40:
                    cropped_img = frame[y1:y2, x1:x2]
                    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    
                    # Chạy mô hình
                    result = self.predictor.predict(cropped_rgb)
                    
                    # Lưu vào lịch sử (tối đa 5 giá trị gần nhất)
                    hist = self.attr_history[track_id]
                    hist["type"].append(result["type"]["class"].lower())
                    hist["color"].append(result["color"]["class"].lower())
                    hist["make"].append(result["make"]["class"].lower())
                    
                    if len(hist["type"]) > 5:
                        hist["type"].pop(0)
                        hist["color"].pop(0)
                        hist["make"].pop(0)
                        
                    # Lấy giá trị xuất hiện nhiều nhất (Majority Vote)
                    best_type = Counter(hist["type"]).most_common(1)[0][0]
                    best_color = Counter(hist["color"]).most_common(1)[0][0]
                    best_make = Counter(hist["make"]).most_common(1)[0][0]
                    
                    self.attr_display[track_id] = f"{best_type} | {best_color} | {best_make}"

            # Hiển thị nhãn
            if track_id in self.attr_display:
                labels.append(f"ID {track_id} | {self.attr_display[track_id]}")
            else:
                labels.append(f"ID {track_id} | ...")
                
        # 4. Vẽ Bounding Box và Label lên Frame
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections, 
            labels=labels
        )
        
        return annotated_frame

    def run(self):
        """Khởi chạy pipeline qua toàn bộ video"""
        print(f"\nBắt đầu xử lý video: {self.video_path}")
        
        video_info = sv.VideoInfo.from_video_path(video_path=self.video_path)
        print(f"Tổng số frame: {video_info.total_frames} | FPS: {video_info.fps}")
        
        sv.process_video(
            source_path=self.video_path,
            target_path=self.output_path,
            callback=self.process_frame
        )
        print(f"\n[HOÀN THÀNH] Video kết quả đã lưu tại: {self.output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn video đầu vào (vd: test.mp4)")
    parser.add_argument("--output", type=str, required=True, help="Đường dẫn video đầu ra (vd: out.mp4)")
    args = parser.parse_args()
    
    pipeline = TrackingPipeline(
        video_path=args.input,
        output_path=args.output
    )
    pipeline.run()
