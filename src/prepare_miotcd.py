from pathlib import Path
import random
import shutil


# =========================
# CẤU HÌNH
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "miotcd"
    / "train1"
)

PROCESSED_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "miotcd"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Để kết quả chia dữ liệu luôn giống nhau
RANDOM_SEED = 42

# Tỷ lệ chia dữ liệu
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =========================
# KIỂM TRA
# =========================

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

random.seed(RANDOM_SEED)


# =========================
# MAPPING CLASS
# =========================

CLASS_MAPPING = {
    "car": "car",
    "bus": "bus",
    "work_van": "van",
    "pickup_truck": "pickup",
    "articulated_truck": "truck",
    "single_unit_truck": "truck"
}


# =========================
# HÀM LẤY ẢNH
# =========================

def get_images(folder):

    return [
        file
        for file in folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]


# =========================
# THU THẬP ẢNH
# =========================

print("=" * 60)
print("CHUẨN BỊ DATASET MIO-TCD")
print("=" * 60)

final_classes = {}

for original_class, final_class in CLASS_MAPPING.items():

    source_folder = RAW_DATASET_PATH / original_class

    images = get_images(source_folder)

    if final_class not in final_classes:
        final_classes[final_class] = []

    final_classes[final_class].extend(images)

    print(
        f"{original_class:25} "
        f"→ {final_class:10} "
        f": {len(images)} ảnh"
    )


# =========================
# TẠO CẤU TRÚC THƯ MỤC
# =========================

splits = ["train", "val", "test"]

for split in splits:
    for class_name in final_classes:

        output_folder = (
            PROCESSED_DATASET_PATH
            / split
            / class_name
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )


# =========================
# CHIA VÀ COPY DỮ LIỆU
# =========================

print("\n")
print("=" * 60)
print("CHIA TRAIN / VALIDATION / TEST")
print("=" * 60)

for class_name, images in final_classes.items():

    # Trộn ngẫu nhiên
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    print(f"\nCLASS: {class_name.upper()}")
    print(f"Tổng: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")

    # Copy ảnh
    for split_name, image_list in split_data.items():

        output_folder = (
            PROCESSED_DATASET_PATH
            / split_name
            / class_name
        )

        for image_path in image_list:

            destination = (
                output_folder
                / image_path.name
            )

            shutil.copy2(
                image_path,
                destination
            )


# =========================
# HOÀN THÀNH
# =========================

print("\n")
print("=" * 60)
print("HOÀN THÀNH CHUẨN BỊ DATASET")
print("=" * 60)

print(
    f"\nDataset đã được tạo tại:\n"
    f"{PROCESSED_DATASET_PATH}"
)