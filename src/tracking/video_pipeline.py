import cv2
import numpy as np
import torch
from pathlib import Path
import supervision as sv
from ultralytics import YOLO
from collections import Counter
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference.predictor import VehiclePredictor

class TrackingPipeline:
    def __init__(
        self,
        video_path: str,
        output_path: str,
        yolo_model_path: str = None,
        color_model_path: str = None,
        make_model_path: str = None
    ):
        self.video_path = video_path
        self.output_path = output_path
        
        # Load mô hình YOLO mới train
        yolo_path = yolo_model_path or str(PROJECT_ROOT / "trained_models" / "best_yolo_vehicle.pt")
        print(f"[1/3] Đang tải YOLO (Detect) từ: {yolo_path}...")
        self.yolo_model = YOLO(yolo_path)
        
        # Model mới chỉ focus vào: 0: car, 1: bus, 2: truck
        self.target_classes = [0, 1, 2] 
        
        print("[2/3] Đang khởi tạo ByteTrack (Tracker)...")
        self.tracker = sv.ByteTrack()
        
        self.box_annotator = sv.BoxAnnotator(thickness=3)
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=1.0,
            text_padding=6,
            text_position=sv.Position.TOP_CENTER
        )
        
        print("[3/3] Đang tải Independent Predictor (Color & Make)...")
        try:
            self.predictor = VehiclePredictor(
                color_model_path=color_model_path,
                make_model_path=make_model_path,
                make_threshold=0.3
            )
            self.predictor_ready = True
        except Exception as e:
            print(f"[CẢNH BÁO] Không tìm thấy model classify: {e}")
            self.predictor_ready = False
            
        self.attr_history = {}
        self.attr_display = {}

    def process_frame(self, frame: np.ndarray, frame_index) -> np.ndarray:
        # Map custom theo dataset.yaml mới train (không dùng COCO nữa)
        CUSTOM_MAP = {0: "Car", 1: "Bus", 2: "Truck"}
        
        results = self.yolo_model(frame, classes=self.target_classes, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        
        labels = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            track_id = detections.tracker_id[i]
            yolo_class_id = int(detections.class_id[i])
            yolo_type = CUSTOM_MAP.get(yolo_class_id, "Unknown")
            
            if track_id not in self.attr_history:
                self.attr_history[track_id] = {"type": [], "color": [], "make": []}
            
            if self.predictor_ready and (frame_index % 3 == 0 or not self.attr_history[track_id]["type"]):
                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if (x2 - x1) > 40 and (y2 - y1) > 40:
                    cropped_img = frame[y1:y2, x1:x2]
                    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    
                    result = self.predictor.predict(cropped_rgb, yolo_type=yolo_type)
                    
                    hist = self.attr_history[track_id]
                    hist["type"].append(result["type"]["class"].capitalize())
                    hist["color"].append(result["color"]["class"].capitalize())
                    hist["make"].append(result["make"]["class"].capitalize())
                    
                    if len(hist["type"]) > 5:
                        hist["type"].pop(0)
                        hist["color"].pop(0)
                        hist["make"].pop(0)
                        
                    best_type = Counter(hist["type"]).most_common(1)[0][0]
                    best_color = Counter(hist["color"]).most_common(1)[0][0]
                    best_make = Counter(hist["make"]).most_common(1)[0][0]
                    
                    if best_make in ["N/a", "N/A", "Unknown"]:
                        self.attr_display[track_id] = f"{best_type} {best_color}"
                    else:
                        self.attr_display[track_id] = f"{best_type} {best_color} {best_make}"

            if track_id in self.attr_display:
                labels.append(f"ID {track_id} {self.attr_display[track_id]}")
            else:
                labels.append(f"ID {track_id} {yolo_type} ...")
                
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        return annotated_frame

    def run(self):
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
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn video đầu vào")
    parser.add_argument("--output", type=str, required=True, help="Đường dẫn video đầu ra")
    args = parser.parse_args()
    
    pipeline = TrackingPipeline(video_path=args.input, output_path=args.output)
    pipeline.run()
