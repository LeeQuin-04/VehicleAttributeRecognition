from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

from detector_tracker import VehicleDetectorTracker
from attribute_provider import VehicleAttributeProvider
from identity_manager import VehicleIdentityManager

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(r"D:\VehicleAttributeRecognition")
INPUT_VIDEO = PROJECT_ROOT / "test_videos" / "test_video_3.mp4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detection_tracking"
OUTPUT_VIDEO = OUTPUT_DIR / "test_video_3_final.mp4"

def get_color(idx: int):
    np.random.seed(idx)
    return tuple(map(int, np.random.randint(0, 255, 3).tolist()))

def draw_vehicle(frame, vehicle, attributes: dict, stable_id: int) -> None:
    x1, y1, x2, y2 = map(int, vehicle.bbox)
    color = get_color(stable_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {stable_id} | {attributes['color']} | {attributes['vehicle_type']} | {attributes['make']}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y1 - 5, text_h + 10)
    cv2.rectangle(frame, (x1, label_y - text_h - 8), (x1 + text_w + 6, label_y + 2), color, -1)
    cv2.putText(frame, label, (x1 + 3, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Không tìm thấy video: {INPUT_VIDEO}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    detector = VehicleDetectorTracker(image_size=800)
    attr_provider = VehicleAttributeProvider()
    identity_manager = VehicleIdentityManager(video_resolution=(w, h), fps=fps)

    print(f"\nĐang xử lý {total_frames} frames...")

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success: break

        tracks = detector.update(frame)
        identity_manager.update(tracks, frame_idx, attr_provider, frame)
        for trk in tracks:
            sid, attrs = identity_manager.get_display_data(trk.track_id)
            if sid:
                draw_vehicle(frame, trk, attrs, sid)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Tiến độ: {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%)", end="\r")

    cap.release()
    writer.release()
    print(f"\nHoàn thành! Video lưu tại: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()