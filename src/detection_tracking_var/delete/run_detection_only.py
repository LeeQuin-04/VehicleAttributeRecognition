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
    / "detection_only"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "test_video_1_detection_only.mp4"
)

MODEL_PATH = "yolo11s.pt"

CONFIDENCE_THRESHOLD = 0.30

IMAGE_SIZE = 640

# COCO:
# 2 = car
# 5 = bus
# 7 = truck
VEHICLE_CLASS_IDS = [2, 5, 7]


# ============================================================
# DRAW
# ============================================================

def draw_detection(
    frame,
    bbox,
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

    label = (
        f"{class_name} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Không tìm thấy video:\n"
            f"{INPUT_VIDEO}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("YOLO DETECTION ONLY")
    print("=" * 60)

    print(
        "Input:",
        INPUT_VIDEO
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Confidence:",
        CONFIDENCE_THRESHOLD
    )

    model = YOLO(
        MODEL_PATH
    )

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Không thể mở video."
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

    while True:

        success, frame = cap.read()

        if not success:
            break

        # ====================================================
        # DETECTION ONLY
        # ====================================================

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            classes=VEHICLE_CLASS_IDS,
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

                confidence = float(
                    boxes.conf[i].item()
                )

                bbox = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                    .astype(float)
                    .tolist()
                )

                draw_detection(
                    frame,
                    bbox,
                    class_name,
                    confidence,
                )

                total_detections += 1

        writer.write(
            frame
        )

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
                f"Processed "
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
        "HOÀN THÀNH DETECTION ONLY"
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
        "Output:",
        OUTPUT_VIDEO
    )


if __name__ == "__main__":
    main()