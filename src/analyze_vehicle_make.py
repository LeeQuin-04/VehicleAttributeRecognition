from pathlib import Path
from PIL import Image
from collections import Counter

# =========================
# ĐƯỜNG DẪN DATASET
# =========================

DATASET_PATH = Path(
    r"D:\cars\car-dataset-200\riotu-cars-dataset-200"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# KHỞI TẠO
# =========================

print("=" * 60)
print("PHÂN TÍCH DATASET VEHICLE MAKE")
print("=" * 60)

brand_counts = {}
model_counts = {}
extension_counts = Counter()
image_sizes = Counter()
corrupted_images = []

total_images = 0

# Lấy các hãng xe
brand_folders = sorted(
    folder for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
)

# =========================
# PHÂN TÍCH TỪNG HÃNG
# =========================

for brand_folder in brand_folders:

    brand_name = brand_folder.name

    print(f"\nĐang kiểm tra hãng: {brand_name}")

    image_files = [
        file
        for file in brand_folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Đếm ảnh
    brand_counts[brand_name] = len(image_files)

    # Đếm số thư mục model / generation
    subfolders = [
        folder
        for folder in brand_folder.rglob("*")
        if folder.is_dir()
    ]

    model_counts[brand_name] = len(subfolders)

    print(f"Số ảnh: {len(image_files)}")
    print(f"Số thư mục con: {len(subfolders)}")

    # Kiểm tra ảnh
    for image_path in image_files:

        try:
            with Image.open(image_path) as img:

                image_sizes[img.size] += 1

                extension_counts[
                    image_path.suffix.lower()
                ] += 1

        except Exception as e:

            corrupted_images.append(
                {
                    "path": str(image_path),
                    "error": str(e)
                }
            )

    total_images += len(image_files)


# =========================
# KẾT QUẢ
# =========================

print("\n")
print("=" * 60)
print("1. SỐ LƯỢNG ẢNH THEO HÃNG")
print("=" * 60)

for brand, count in brand_counts.items():
    print(f"{brand:15}: {count} ảnh")

print(f"\nTỔNG SỐ ẢNH: {total_images}")


print("\n")
print("=" * 60)
print("2. SỐ THƯ MỤC CON THEO HÃNG")
print("=" * 60)

for brand, count in model_counts.items():
    print(f"{brand:15}: {count}")


print("\n")
print("=" * 60)
print("3. ĐỊNH DẠNG ẢNH")
print("=" * 60)

for extension, count in extension_counts.items():
    print(f"{extension:10}: {count}")


print("\n")
print("=" * 60)
print("4. 10 KÍCH THƯỚC ẢNH PHỔ BIẾN NHẤT")
print("=" * 60)

for size, count in image_sizes.most_common(10):
    print(f"{size}: {count} ảnh")


print("\n")
print("=" * 60)
print("5. ẢNH BỊ LỖI")
print("=" * 60)

if not corrupted_images:
    print("Không phát hiện ảnh bị lỗi.")
else:
    print(f"Phát hiện {len(corrupted_images)} ảnh bị lỗi.\n")

    for item in corrupted_images[:10]:
        print(item["path"])
        print("Lỗi:", item["error"])
        print("-" * 40)


print("\n")
print("=" * 60)
print("HOÀN THÀNH PHÂN TÍCH DATASET")
print("=" * 60)