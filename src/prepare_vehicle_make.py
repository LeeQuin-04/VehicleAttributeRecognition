from pathlib import Path
import random
import shutil

# =========================
# CẤU HÌNH
# =========================

PROJECT_ROOT = Path(r"D:\VehicleAttributeRecognition")

DATASET_PATH = Path(r"D:\cars\car-dataset-200\riotu-cars-dataset-200")

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "vehicle_make"

# Các định dạng ảnh được chấp nhận
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Các hãng được giữ lại
SELECTED_MAKES = [
    "audi", "bmw", "cadillac", "chevrolet", "dodge",
    "ford", "gmc", "honda", "hyundai", "infiniti",
    "jeep", "kia", "landrover", "lexus", "mazda",
    "mercedes", "mitsubishi", "nissan", "porsche", "toyota"
]

# Tỷ lệ chia dataset
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Seed để kết quả chia luôn giống nhau
RANDOM_SEED = 42


# =========================
# KHỞI TẠO
# =========================

random.seed(RANDOM_SEED)

print("=" * 60)
print("CHUẨN BỊ DATASET VEHICLE MAKE")
print("=" * 60)


# =========================
# KIỂM TRA DATASET (Đã sửa biến RAW_DATASET_PATH -> DATASET_PATH)
# =========================

if not DATASET_PATH.exists():
    print("\nKhông tìm thấy dataset tại:")
    print(DATASET_PATH)
    raise SystemExit


# =========================
# TẠO THƯ MỤC OUTPUT
# =========================

for split in ["train", "val", "test"]:
    split_path = OUTPUT_PATH / split
    split_path.mkdir(parents=True, exist_ok=True)


# =========================
# XỬ LÝ TỪNG HÃNG
# =========================

dataset_summary = {}

for make_name in SELECTED_MAKES:
    make_path = DATASET_PATH / make_name

    if not make_path.exists():
        print(f"\n⚠ Không tìm thấy hãng: {make_name}")
        continue

    print("\n" + "=" * 60)
    print(f"ĐANG XỬ LÝ: {make_name.upper()}")
    print("=" * 60)

    # Lấy toàn bộ ảnh trong các thư mục con
    image_files = [
        file for file in make_path.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    total_images = len(image_files)
    print(f"Tổng số ảnh: {total_images}")

    if total_images == 0:
        print(f"⚠ Không có ảnh hợp lệ nào cho hãng {make_name}")
        continue

    # Trộn ảnh
    random.shuffle(image_files)

    # Tính số lượng
    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    splits_data = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

    print(f"Train: {len(splits_data['train'])}")
    print(f"Validation: {len(splits_data['val'])}")
    print(f"Test: {len(splits_data['test'])}")

    # =========================
    # COPY VÀ ĐỔI TÊN FILE (Tối ưu hóa vòng lặp)
    # =========================
    for split_name, files in splits_data.items():
        split_make_dir = OUTPUT_PATH / split_name / make_name
        split_make_dir.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(files):
            extension = image_path.suffix.lower()
            new_name = f"{make_name}_{split_name}_{index:06d}{extension}"
            destination = split_make_dir / new_name

            shutil.copy2(image_path, destination)

    # Lưu thống kê
    dataset_summary[make_name] = {
        "total": total_images,
        "train": len(splits_data["train"]),
        "val": len(splits_data["val"]),
        "test": len(splits_data["test"])
    }


# =========================
# KẾT QUẢ
# =========================

print("\n")
print("=" * 60)
print("TỔNG KẾT DATASET")
print("=" * 60)

print(
    f"{'HÃNG':15}"
    f"{'TRAIN':>10}"
    f"{'VAL':>10}"
    f"{'TEST':>10}"
    f"{'TOTAL':>10}"
)

print("-" * 55)

total_train = 0
total_val = 0
total_test = 0
total_all = 0

for make_name, data in dataset_summary.items():
    print(
        f"{make_name:15}"
        f"{data['train']:>10}"
        f"{data['val']:>10}"
        f"{data['test']:>10}"
        f"{data['total']:>10}"
    )

    total_train += data["train"]
    total_val += data["val"]
    total_test += data["test"]
    total_all += data["total"]

print("-" * 55)

print(
    f"{'TOTAL':15}"
    f"{total_train:>10}"
    f"{total_val:>10}"
    f"{total_test:>10}"
    f"{total_all:>10}"
)

print("\n")
print("=" * 60)
print("HOÀN THÀNH CHUẨN BỊ DATASET")
print("=" * 60)

print("\nDataset đã được tạo tại:")
print(OUTPUT_PATH)