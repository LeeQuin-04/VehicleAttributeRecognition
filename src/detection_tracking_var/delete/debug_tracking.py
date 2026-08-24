from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Optional

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
    / "tracking_debug_v2"
)

MODEL_PATH = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.30

IMAGE_SIZE = 640

TRACKER = "bytetrack.yaml"

VEHICLE_CLASSES = {
    "car",
    "bus",
    "truck",
}


# ============================================================
# DEBUG PARAMETERS
# ============================================================

# Bỏ qua giai đoạn đầu để tracker ổn định.
WARMUP_FRAMES = 50

# Số frame tối đa giữa lúc ID cũ mất
# và ID mới xuất hiện.
MAX_GAP_FRAMES = 10

# IoU tối thiểu để coi hai bbox có thể là cùng object.
MIN_IOU = 0.15

# Khoảng cách tâm tối đa.
MAX_CENTER_DISTANCE = 100

# Tối đa số sự kiện lưu lại.
MAX_SWITCH_EVENTS = 20

# Số frame trước sự kiện để lưu.
HISTORY_FRAMES = 20


# ============================================================
# GEOMETRY
# ============================================================

def get_center(bbox: list[float]):
    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def center_distance(
    bbox_a: list[float],
    bbox_b: list[float],
) -> float:

    ax, ay = get_center(bbox_a)
    bx, by = get_center(bbox_b)

    return (
        (ax - bx) ** 2
        + (ay - by) ** 2
    ) ** 0.5


def calculate_iou(
    box_a: list[float],
    box_b: list[float],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(
        ax1,
        bx1
    )

    inter_y1 = max(
        ay1,
        by1
    )

    inter_x2 = min(
        ax2,
        bx2
    )

    inter_y2 = min(
        ay2,
        by2
    )

    inter_width = max(
        0.0,
        inter_x2 - inter_x1
    )

    inter_height = max(
        0.0,
        inter_y2 - inter_y1
    )

    intersection = (
        inter_width
        * inter_height
    )

    area_a = max(
        0.0,
        ax2 - ax1
    ) * max(
        0.0,
        ay2 - ay1
    )

    area_b = max(
        0.0,
        bx2 - bx1
    ) * max(
        0.0,
        by2 - by1
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# DRAW
# ============================================================

def draw_box(
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

    label = (
        f"ID {track_id} | "
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


def draw_debug_message(
    frame,
    message,
):
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 45),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        message,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():

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
    print("TRACKING DEBUG V2")
    print("=" * 60)

    print("Input:", INPUT_VIDEO)
    print("Tracker:", TRACKER)
    print(
        "Confidence:",
        CONFIDENCE_THRESHOLD
    )
    print(
        "Min IoU:",
        MIN_IOU
    )
    print(
        "Max center distance:",
        MAX_CENTER_DISTANCE
    )
    print(
        "Max gap:",
        MAX_GAP_FRAMES
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # --------------------------------------------------------
    # Lưu frame gần nhất
    # --------------------------------------------------------

    frame_history = deque(
        maxlen=HISTORY_FRAMES
    )

    # --------------------------------------------------------
    # ID đang active:
    #
    # {
    #   track_id: {
    #       bbox,
    #       class_name,
    #       confidence
    #   }
    # }
    # --------------------------------------------------------

    active_tracks = {}

    # --------------------------------------------------------
    # ID đã mất:
    #
    # {
    #   track_id: {
    #       last_frame,
    #       bbox,
    #       class_name,
    #       confidence
    #   }
    # }
    # --------------------------------------------------------

    lost_tracks = {}

    frame_index = 0
    switch_event_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # ====================================================
        # YOLO + BYTETRACK
        # ====================================================

        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            classes=[2, 5, 7],
            verbose=False,
        )

        result = results[0]

        current_tracks = {}

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(
                len(boxes)
            ):

                if boxes.id is None:
                    continue

                class_id = int(
                    boxes.cls[i].item()
                )

                class_name = (
                    model.names[class_id]
                )

                if class_name not in (
                    VEHICLE_CLASSES
                ):
                    continue

                track_id = int(
                    boxes.id[i].item()
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

                current_tracks[
                    track_id
                ] = {
                    "bbox": bbox,
                    "class_name": class_name,
                    "confidence": confidence,
                }

                draw_box(
                    frame,
                    bbox,
                    track_id,
                    class_name,
                    confidence,
                )

        # ====================================================
        # BỎ QUA WARMUP
        # ====================================================

        if frame_index < WARMUP_FRAMES:

            active_tracks = (
                current_tracks
            )

            frame_history.append(
                (
                    frame_index,
                    frame.copy()
                )
            )

            frame_index += 1

            continue

        # ====================================================
        # TRACK BỊ MẤT
        # ====================================================

        disappeared_ids = (
            set(active_tracks.keys())
            - set(current_tracks.keys())
        )

        for old_id in disappeared_ids:

            old_data = (
                active_tracks[old_id]
            )

            lost_tracks[old_id] = {
                "last_frame": frame_index,
                "bbox": old_data["bbox"],
                "class_name": old_data[
                    "class_name"
                ],
                "confidence": old_data[
                    "confidence"
                ],
            }

        # ====================================================
        # TÌM NGHI VẤN ID SWITCH
        # ====================================================

        suspected_switch = False

        for new_id, new_data in (
            current_tracks.items()
        ):

            for old_id, old_data in list(
                lost_tracks.items()
            ):

                if old_id == new_id:
                    continue

                gap = (
                    frame_index
                    - old_data["last_frame"]
                )

                if gap < 1:
                    continue

                if gap > MAX_GAP_FRAMES:
                    continue

                # Phải cùng class.
                if (
                    old_data["class_name"]
                    != new_data["class_name"]
                ):
                    continue

                iou = calculate_iou(
                    old_data["bbox"],
                    new_data["bbox"],
                )

                distance = (
                    center_distance(
                        old_data["bbox"],
                        new_data["bbox"],
                    )
                )

                # ------------------------------------------------
                # Điều kiện nghi ngờ:
                #
                # IoU cao
                # HOẶC
                # vị trí gần + có overlap
                # ------------------------------------------------

                is_candidate = (
                    iou >= MIN_IOU
                    or (
                        distance
                        <= MAX_CENTER_DISTANCE
                        and iou > 0
                    )
                )

                if not is_candidate:
                    continue

                # ------------------------------------------------
                # THIẾT LẬP EVENT
                # ------------------------------------------------

                switch_event_count += 1
                suspected_switch = True

                if (
                    switch_event_count
                    <= MAX_SWITCH_EVENTS
                ):

                    event_dir = (
                        OUTPUT_DIR
                        / (
                            f"switch_"
                            f"{switch_event_count:03d}"
                        )
                    )

                    event_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    print(
                        "\n"
                        + "!" * 60
                    )

                    print(
                        "SUSPECTED ID SWITCH"
                    )

                    print(
                        "Frame:",
                        frame_index
                    )

                    print(
                        "Old ID:",
                        old_id
                    )

                    print(
                        "New ID:",
                        new_id
                    )

                    print(
                        "Gap:",
                        gap
                    )

                    print(
                        "IoU:",
                        f"{iou:.3f}"
                    )

                    print(
                        "Center distance:",
                        f"{distance:.2f}"
                    )

                    print(
                        "Class:",
                        new_data["class_name"]
                    )

                    print(
                        "!" * 60
                    )

                    # --------------------------------------------
                    # BEFORE
                    # --------------------------------------------

                    if len(frame_history) > 0:

                        before_frame_index, before_frame = (
                            frame_history[-1]
                        )

                        before_frame = (
                            before_frame.copy()
                        )

                        draw_debug_message(
                            before_frame,
                            (
                                f"BEFORE | "
                                f"ID {old_id}"
                            ),
                        )

                        before_path = (
                            event_dir
                            / (
                                "before_"
                                f"{before_frame_index}.jpg"
                            )
                        )

                        cv2.imwrite(
                            str(before_path),
                            before_frame
                        )

                    # --------------------------------------------
                    # AFTER
                    # --------------------------------------------

                    after_frame = (
                        frame.copy()
                    )

                    draw_debug_message(
                        after_frame,
                        (
                            f"AFTER | "
                            f"OLD {old_id} "
                            f"-> NEW {new_id}"
                        ),
                    )

                    after_path = (
                        event_dir
                        / (
                            "after_"
                            f"{frame_index}.jpg"
                        )
                    )

                    cv2.imwrite(
                        str(after_path),
                        after_frame
                    )

                # Không cần tiếp tục so ID cũ này
                # với các ID mới khác trong cùng frame.
                del lost_tracks[
                    old_id
                ]

                break

            if (
                suspected_switch
                and new_id in current_tracks
            ):
                # Chỉ cần xử lý một event
                # đáng nghi cho new_id.
                continue

        # ====================================================
        # XÓA TRACK ĐÃ MẤT QUÁ LÂU
        # ====================================================

        for old_id in list(
            lost_tracks.keys()
        ):

            age = (
                frame_index
                - lost_tracks[old_id][
                    "last_frame"
                ]
            )

            if age > MAX_GAP_FRAMES:

                del lost_tracks[
                    old_id
                ]

        # ====================================================
        # UPDATE
        # ====================================================

        active_tracks = (
            current_tracks
        )

        frame_history.append(
            (
                frame_index,
                frame.copy()
            )
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

    print(
        "\n"
        + "=" * 60
    )

    print(
        "DEBUG V2 HOÀN THÀNH"
    )

    print(
        "=" * 60
    )

    print(
        "Frames:",
        frame_index
    )

    print(
        "Suspected switch events:",
        min(
            switch_event_count,
            MAX_SWITCH_EVENTS
        )
    )

    print(
        "Output:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()