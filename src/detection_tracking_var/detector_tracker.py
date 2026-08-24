import os
from pathlib import Path
from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class TrackedVehicle:
    track_id: int
    bbox: list[float]
    confidence: float
    detector_class: str

class VehicleDetectorTracker:
    VEHICLE_CLASS_IDS = [2, 5, 7] 

    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        confidence_threshold: float = 0.25,
        image_size: int = 800, # ĐIỂM CÂN BẰNG: 800px
        tracker_config: str = "custom_tracker.yaml",
        max_det: int = 40
    ) -> None:
        self.device = "cpu"
            
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.max_det = max_det
        
        current_dir = Path(__file__).parent.resolve()
        tracker_path = current_dir / tracker_config
        if not tracker_path.exists():
            raise FileNotFoundError(f"Không tìm thấy cấu hình: {tracker_path}")
        self.tracker = str(tracker_path)

    def update(self, frame) -> list[TrackedVehicle]:
        img_h, img_w = frame.shape[:2]
        min_area_threshold = (img_w * img_h) * 0.0004

        results = self.model.track(
            source=frame, persist=True, tracker=self.tracker,
            conf=self.confidence_threshold, iou=0.45, 
            imgsz=self.image_size, classes=self.VEHICLE_CLASS_IDS, verbose=False,
            device=self.device
        )

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []

        boxes = results[0].boxes
        final_tracks = []
        for i in range(len(boxes)):
            tid = int(boxes.id[i].item())
            bbox = boxes.xyxy[i].cpu().numpy().flatten().astype(float).tolist()
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if (w * h) < min_area_threshold: continue

            cls_name = self.model.names[int(boxes.cls[i].item())].lower()
            final_tracks.append(TrackedVehicle(tid, bbox, float(boxes.conf[i].item()), cls_name))
        return final_tracks