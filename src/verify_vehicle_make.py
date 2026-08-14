from pathlib import Path
from PIL import Image
from collections import Counter

# =========================
# CẤU HÌNH
# =========================

PROJECT_ROOT = Path(r"D:\VehicleAttributeRecognition")

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vehicle_make"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# =========================
# KIỂM TRA DATASET
# =========================

print("=" * 60)
print("KIỂM TRA DATASET VEHICLE MAKE")
print("=" * 60)

total_all = 0

for split_name in ["train", "val", "test"]:

    split_path = DATASET_PATH / split_name

    print("\n")
    print("=" * 60)
    print(f"KIỂM TRA: {split_name.upper()}")
    print("=" * 60)

    if not split_path.exists():
        print(f"❌ Không tìm thấy thư mục: {split_path}")
        continue

    class_folders = sorted(
        folder
        for folder in split_path.iterdir()
        if folder.is_dir()
    )

    split_total = 0

    for class_folder in class_folders:

        image_files = [
            file
            for file in class_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        count = len(image_files)

        print(f"{class_folder.name:15} : {count} ảnh")

        split_total += count

    total_all += split_total

    print("-" * 60)
    print(f"TỔNG {split_name.upper()}: {split_total} ảnh")


# =========================
# KIỂM TRA ẢNH LỖI
# =========================

print("\n")
print("=" * 60)
print("KIỂM TRA ẢNH BỊ LỖI")
print("=" * 60)

corrupted_images = []

for image_path in DATASET_PATH.rglob("*"):

    if (
        image_path.is_file()
        and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ):

        try:
            with Image.open(image_path) as img:
                img.verify()

        except Exception as e:
            corrupted_images.append(
                (image_path, str(e))
            )


if not corrupted_images:
    print("✓ Không phát hiện ảnh bị lỗi.")
else:
    print(
        f"❌ Phát hiện {len(corrupted_images)} ảnh bị lỗi:"
    )

    for image_path, error in corrupted_images[:10]:
        print(image_path)
        print(error)
        print("-" * 40)


# =========================
# TỔNG KẾT
# =========================

print("\n")
print("=" * 60)
print("HOÀN THÀNH KIỂM TRA")
print("=" * 60)

print(f"\nTỔNG SỐ ẢNH TOÀN DATASET: {total_all}")