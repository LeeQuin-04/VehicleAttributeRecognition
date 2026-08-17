from tracker import (
    Detection,
    VehicleTracker,
)


def main() -> None:

    tracker = VehicleTracker()

    # FRAME 1

    frame_1 = [
        Detection(
            bbox=[100, 100, 200, 200],
            confidence=0.95,
            class_name="car",
        ),
        Detection(
            bbox=[400, 100, 500, 200],
            confidence=0.90,
            class_name="truck",
        ),
    ]

    tracked_1 = tracker.update(
        frame_1
    )

    print("FRAME 1")

    for vehicle in tracked_1:
        print(vehicle)

    # FRAME 2

    frame_2 = [
        Detection(
            bbox=[110, 100, 210, 200],
            confidence=0.94,
            class_name="car",
        ),
        Detection(
            bbox=[405, 105, 505, 205],
            confidence=0.91,
            class_name="truck",
        ),
    ]

    tracked_2 = tracker.update(
        frame_2
    )

    print("\nFRAME 2")

    for vehicle in tracked_2:
        print(vehicle)

    # FRAME 3

    frame_3 = [
        Detection(
            bbox=[120, 105, 220, 205],
            confidence=0.96,
            class_name="car",
        ),
        Detection(
            bbox=[410, 110, 510, 210],
            confidence=0.89,
            class_name="truck",
        ),
    ]

    tracked_3 = tracker.update(
        frame_3
    )

    print("\nFRAME 3")

    for vehicle in tracked_3:
        print(vehicle)


if __name__ == "__main__":
    main()