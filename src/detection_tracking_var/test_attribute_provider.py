from __future__ import annotations

from pathlib import Path
import cv2

from attribute_provider import VehicleAttributeProvider

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_IMAGE = (
    PROJECT_ROOT
    / "test_videos"
    / "test_car.png"
)


def main() -> None:

    if not TEST_IMAGE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ảnh:\n{TEST_IMAGE}"
        )

    image = cv2.imread(
        str(TEST_IMAGE)
    )

    if image is None:
        raise RuntimeError(
            "Không thể đọc ảnh."
        )

    print("=" * 60)
    print("ĐANG LOAD 3 ATTRIBUTE MODELS")
    print("=" * 60)

    provider = VehicleAttributeProvider()

    print("\n" + "=" * 60)
    print("ĐANG NHẬN DIỆN")
    print("=" * 60)

    attributes = provider.predict(
        image
    )

    print()

    print(
        f"Color: "
        f"{attributes['color']} "
        f"({attributes['color_confidence']:.2%})"
    )

    print(
        f"Type: "
        f"{attributes['vehicle_type']} "
        f"({attributes['type_confidence']:.2%})"
    )

    print(
        f"Make: "
        f"{attributes['make']} "
        f"({attributes['make_confidence']:.2%})"
    )


if __name__ == "__main__":
    main()