from pathlib import Path
from PIL import Image
from collections import Counter

# =========================
# ĐƯỜNG DẪN DATASET
# =========================

DATASET_PATH = Path(
    r"D:\VehicleAttributeRecognition\data\raw\miotcd\train1"
)

# Các định dạng ảnh được chấp nhận
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# =========================
# THỐNG KÊ
# =========================

print("=" * 60)
print("PHÂN TÍCH DATASET MIO-TCD")
print("=" * 60)

class_counts = {}
image_sizes = Counter()
extension_counts = Counter()
corrupted_images = []

# Lấy danh sách các folder class
class_folders = sorted(
    [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)

total_images = 0

for class_folder in class_folders:

    class_name = class_folder.name

    image_files = [
        file for file in class_folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    class_counts[class_name] = len(image_files)

    print(f"\nĐang kiểm tra class: {class_name}")
    print(f"Số ảnh: {len(image_files)}")

    # Kiểm tra từng ảnh
    for image_path in image_files:

        try:
            with Image.open(image_path) as img:

                # Kích thước ảnh
                image_sizes[img.size] += 1

                # Định dạng file
                extension_counts[image_path.suffix.lower()] += 1

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
print("1. SỐ LƯỢNG ẢNH THEO CLASS")
print("=" * 60)

for class_name, count in class_counts.items():
    print(f"{class_name:30} : {count}")

print(f"\nTỔNG SỐ ẢNH: {total_images}")


print("\n")
print("=" * 60)
print("2. ĐỊNH DẠNG ẢNH")
print("=" * 60)

for extension, count in extension_counts.items():
    print(f"{extension:10} : {count}")


print("\n")
print("=" * 60)
print("3. 10 KÍCH THƯỚC ẢNH PHỔ BIẾN NHẤT")
print("=" * 60)

for size, count in image_sizes.most_common(10):
    print(f"{size} : {count} ảnh")


print("\n")
print("=" * 60)
print("4. ẢNH BỊ LỖI")
print("=" * 60)

if len(corrupted_images) == 0:
    print("Không phát hiện ảnh bị lỗi.")
else:
    print(f"Phát hiện {len(corrupted_images)} ảnh bị lỗi:\n")

    for item in corrupted_images[:10]:
        print(item["path"])
        print("Lỗi:", item["error"])
        print("-" * 40)


print("\n")
print("=" * 60)
print("HOÀN THÀNH PHÂN TÍCH DATASET")
print("=" * 60)