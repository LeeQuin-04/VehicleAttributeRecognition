from pathlib import Path
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import random

# =========================
# ĐƯỜNG DẪN PROJECT
# =========================

PROJECT_ROOT = Path(
    r"D:\VehicleAttributeRecognition"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vehicle_color"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "results"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

# =========================
# CẤU HÌNH
# =========================

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

# =========================
# BẮT ĐẦU PHÂN TÍCH
# =========================

print("=" * 60)
print("PHÂN TÍCH DATASET VEHICLE COLOR")
print("=" * 60)

split_class_counts = {}
total_class_counts = Counter()
extension_counts = Counter()
image_sizes = Counter()
corrupted_images = []

# =========================
# KIỂM TRA TỪNG SPLIT
# =========================

for split in SPLITS:

    split_path = DATASET_PATH / split

    print("\n")
    print("=" * 60)
    print(f"KIỂM TRA: {split.upper()}")
    print("=" * 60)

    if not split_path.exists():
        print(f"Không tìm thấy thư mục: {split_path}")
        continue

    class_folders = sorted(
        [
            folder
            for folder in split_path.iterdir()
            if folder.is_dir()
        ]
    )

    split_class_counts[split] = {}

    for class_folder in class_folders:

        class_name = class_folder.name

        image_files = [
            file
            for file in class_folder.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        split_class_counts[split][class_name] = len(image_files)

        total_class_counts[class_name] += len(image_files)

        print(
            f"{class_name:15}: "
            f"{len(image_files)} ảnh"
        )

        # Kiểm tra từng ảnh
        for image_path in image_files:

            try:

                with Image.open(image_path) as img:

                    # Kiểm tra kích thước
                    image_sizes[img.size] += 1

                    # Định dạng
                    extension_counts[
                        image_path.suffix.lower()
                    ] += 1

                    # Kiểm tra ảnh có thể đọc hoàn toàn
                    img.verify()

            except Exception as e:

                corrupted_images.append(
                    {
                        "path": str(image_path),
                        "error": str(e)
                    }
                )


# =========================
# TỔNG HỢP CLASS
# =========================

print("\n")
print("=" * 60)
print("TỔNG SỐ ẢNH THEO MÀU")
print("=" * 60)

for class_name, count in sorted(
    total_class_counts.items()
):

    print(
        f"{class_name:15}: {count} ảnh"
    )

total_images = sum(
    total_class_counts.values()
)

print("\n")
print(f"TỔNG SỐ ẢNH: {total_images}")

print("\n")
print("=" * 60)
print("PHÂN BỐ TRAIN / VAL / TEST")
print("=" * 60)

all_classes = sorted(
    total_class_counts.keys()
)

print(
    f"{'COLOR':15}"
    f"{'TRAIN':>10}"
    f"{'VAL':>10}"
    f"{'TEST':>10}"
    f"{'TOTAL':>10}"
)

print("-" * 55)

for class_name in all_classes:

    train_count = split_class_counts.get(
        "train",
        {}
    ).get(
        class_name,
        0
    )

    val_count = split_class_counts.get(
        "val",
        {}
    ).get(
        class_name,
        0
    )

    test_count = split_class_counts.get(
        "test",
        {}
    ).get(
        class_name,
        0
    )

    total = (
        train_count
        + val_count
        + test_count
    )

    print(
        f"{class_name:15}"
        f"{train_count:>10}"
        f"{val_count:>10}"
        f"{test_count:>10}"
        f"{total:>10}"
    )


# =========================
# ĐỊNH DẠNG ẢNH
# =========================

print("\n")
print("=" * 60)
print("ĐỊNH DẠNG ẢNH")
print("=" * 60)

for extension, count in extension_counts.items():

    print(
        f"{extension:10}: {count}"
    )


# =========================
# KÍCH THƯỚC PHỔ BIẾN
# =========================

print("\n")
print("=" * 60)
print("10 KÍCH THƯỚC ẢNH PHỔ BIẾN NHẤT")
print("=" * 60)

for size, count in image_sizes.most_common(10):

    print(
        f"{size}: {count} ảnh"
    )


# =========================
# ẢNH LỖI
# =========================

print("\n")
print("=" * 60)
print("ẢNH BỊ LỖI")
print("=" * 60)

if len(corrupted_images) == 0:

    print("Không phát hiện ảnh bị lỗi.")

else:

    print(
        f"Phát hiện "
        f"{len(corrupted_images)} ảnh bị lỗi"
    )

    for item in corrupted_images[:10]:

        print("\nFile:")
        print(item["path"])

        print("Lỗi:")
        print(item["error"])


# =========================
# HIỂN THỊ ẢNH MẪU
# =========================

print("\n")
print("=" * 60)
print("TẠO ẢNH MẪU")
print("=" * 60)

TRAIN_PATH = (
    DATASET_PATH
    / "train"
)

class_folders = sorted(
    [
        folder
        for folder in TRAIN_PATH.iterdir()
        if folder.is_dir()
    ]
)

fig, axes = plt.subplots(
    3,
    5,
    figsize=(16, 10)
)

axes = axes.flatten()

for i, class_folder in enumerate(class_folders):

    class_name = class_folder.name

    image_files = [
        file
        for file in class_folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(image_files) == 0:

        axes[i].axis("off")
        continue

    image_path = random.choice(
        image_files
    )

    try:

        with Image.open(image_path) as img:

            axes[i].imshow(
                img.convert("RGB")
            )

    except Exception:

        axes[i].text(
            0.5,
            0.5,
            "Error",
            ha="center",
            va="center"
        )

    axes[i].set_title(
        class_name
    )

    axes[i].axis("off")


plt.tight_layout()

sample_output = (
    OUTPUT_PATH
    / "vehicle_color_samples.png"
)

plt.savefig(
    sample_output,
    dpi=150
)

print("Đã lưu ảnh mẫu tại:")
print(sample_output)

plt.show()


# =========================
# HOÀN THÀNH
# =========================

print("\n")
print("=" * 60)
print("HOÀN THÀNH PHÂN TÍCH DATASET")
print("=" * 60)