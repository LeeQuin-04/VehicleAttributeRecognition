from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"D:\VehicleAttributeRecognition"
)

INPUT_VIDEO = (
    PROJECT_ROOT
    / "test_videos"
    / "test_video_1.mp4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "detection_tracking"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "test_video_1_botsort_reid.mp4"
)

MODEL_PATH = "yolo11n.pt"

TRACKER_CONFIG = (
    PROJECT_ROOT
    / "src"
    / "detection_tracking"
    / "configs"
    / "botsort_vehicle_reid.yaml"
)

CONFIDENCE_THRESHOLD = 0.30
IMAGE_SIZE = 640

VEHICLE_CLASSES = {
    "car",
    "bus",
    "truck",
}


def draw_vehicle(
    frame,
    bbox,
    track_id,
    class_name,
    confidence,
):
    x1, y1, x2, y2 = map(
        int,
        bbox
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    if track_id is not None:
        label = (
            f"ID {track_id} | "
            f"{class_name} "
            f"{confidence:.2f}"
        )
    else:
        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

    text_y = max(
        25,
        y1 - 10
    )

    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Không tìm thấy video:\n"
            f"{INPUT_VIDEO}"
        )

    if not TRACKER_CONFIG.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tracker config:\n"
            f"{TRACKER_CONFIG}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("MODULE 1 - BOt-SORT + ReID")
    print("=" * 60)

    print("\nĐang load YOLO...")

    model = YOLO(
        MODEL_PATH
    )

    print(
        "✓ YOLO loaded."
    )

    print(
        "Tracker:",
        TRACKER_CONFIG
    )

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Không thể mở video:\n"
            f"{INPUT_VIDEO}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Không thể tạo output video."
        )

    frame_index = 0
    total_detections = 0
    unique_track_ids = set()

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            source=frame,
            persist=True,
            tracker=str(
                TRACKER_CONFIG
            ),
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(
                len(boxes)
            ):

                class_id = int(
                    boxes.cls[i].item()
                )

                class_name = (
                    model.names[class_id]
                )

                if class_name not in VEHICLE_CLASSES:
                    continue

                confidence = float(
                    boxes.conf[i].item()
                )

                bbox = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )

                track_id = None

                if boxes.id is not None:
                    track_id = int(
                        boxes.id[i].item()
                    )

                    unique_track_ids.add(
                        track_id
                    )

                draw_vehicle(
                    frame,
                    bbox,
                    track_id,
                    class_name,
                    confidence,
                )

                total_detections += 1

        writer.write(frame)

        frame_index += 1

        if frame_index % 100 == 0:
            progress = (
                frame_index
                / total_frames
                * 100
                if total_frames > 0
                else 0
            )

            print(
                f"Processed: "
                f"{frame_index}/"
                f"{total_frames} "
                f"({progress:.1f}%)"
            )

    cap.release()
    writer.release()

    print(
        "\n" + "=" * 60
    )

    print(
        "HOÀN THÀNH BoT-SORT + ReID"
    )

    print(
        "=" * 60
    )

    print(
        "Frames:",
        frame_index
    )

    print(
        "Total detections:",
        total_detections
    )

    print(
        "Unique Track IDs:",
        len(unique_track_ids)
    )

    print(
        "Output:",
        OUTPUT_VIDEO
    )


if __name__ == "__main__":
    main()