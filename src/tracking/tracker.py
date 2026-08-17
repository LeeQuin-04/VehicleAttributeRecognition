from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import supervision as sv


@dataclass
class Detection:
    bbox: list[float]
    confidence: float
    class_name: str


@dataclass
class TrackedVehicle:
    track_id: int
    bbox: list[float]
    confidence: float
    class_name: str


class VehicleTracker:
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
    ) -> None:

        self.tracker = sv.ByteTrack(
            track_activation_threshold=(
                track_activation_threshold
            ),
            lost_track_buffer=(
                lost_track_buffer
            ),
            minimum_matching_threshold=(
                minimum_matching_threshold
            ),
            minimum_consecutive_frames=(
                minimum_consecutive_frames
            ),
        )

    def update(
        self,
        detections: Iterable[Detection],
    ) -> list[TrackedVehicle]:

        detections = list(detections)

        if not detections:
            return []

        xyxy = np.array(
            [d.bbox for d in detections],
            dtype=np.float32,
        )

        confidence = np.array(
            [d.confidence for d in detections],
            dtype=np.float32,
        )

        class_names = [
            d.class_name
            for d in detections
        ]

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
        )

        tracked = self.tracker.update_with_detections(
            sv_detections
        )

        results: list[TrackedVehicle] = []

        for i in range(len(tracked)):

            track_id = int(
                tracked.tracker_id[i]
            )

            bbox = (
                tracked.xyxy[i]
                .astype(float)
                .tolist()
            )

            conf = float(
                tracked.confidence[i]
            )

            # Tìm class gần nhất theo bbox.
            matched_class = "unknown"

            if len(detections) > 0:
                best_index = self._find_best_match(
                    tracked.xyxy[i],
                    xyxy,
                )

                if best_index is not None:
                    matched_class = class_names[
                        best_index
                    ]

            results.append(
                TrackedVehicle(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_name=matched_class,
                )
            )

        return results

    @staticmethod
    def _find_best_match(
        target_bbox: np.ndarray,
        candidate_boxes: np.ndarray,
    ) -> int | None:

        if len(candidate_boxes) == 0:
            return None

        ious = []

        for candidate in candidate_boxes:
            ious.append(
                VehicleTracker._iou(
                    target_bbox,
                    candidate,
                )
            )

        return int(
            np.argmax(ious)
        )

    @staticmethod
    def _iou(
        box_a: np.ndarray,
        box_b: np.ndarray,
    ) -> float:

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(
            0.0,
            inter_x2 - inter_x1,
        )

        inter_h = max(
            0.0,
            inter_y2 - inter_y1,
        )

        inter_area = (
            inter_w * inter_h
        )

        area_a = max(
            0.0,
            ax2 - ax1,
        ) * max(
            0.0,
            ay2 - ay1,
        )

        area_b = max(
            0.0,
            bx2 - bx1,
        ) * max(
            0.0,
            by2 - by1,
        )

        union = (
            area_a
            + area_b
            - inter_area
        )

        if union <= 0:
            return 0.0

        return inter_area / union