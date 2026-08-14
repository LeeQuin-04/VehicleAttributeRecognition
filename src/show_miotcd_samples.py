from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

# =========================
# ĐƯỜNG DẪN
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "miotcd"
    / "train1"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "results"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# =========================
# LẤY DANH SÁCH CLASS
# =========================

class_folders = sorted(
    [
        folder
        for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    ]
)

# =========================
# TẠO ẢNH MẪU
# =========================

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

axes = axes.flatten()

for i, class_folder in enumerate(class_folders):

    class_name = class_folder.name

    image_files = [
        file
        for file in class_folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Chọn ngẫu nhiên 1 ảnh
    image_path = random.choice(image_files)

    # Hiển thị ảnh
    with Image.open(image_path) as img:
        axes[i].imshow(img)

    axes[i].set_title(class_name)
    axes[i].axis("off")


# Ẩn ô thừa
for j in range(len(class_folders), len(axes)):
    axes[j].axis("off")


plt.tight_layout()

# =========================
# LƯU KẾT QUẢ
# =========================

sample_output = OUTPUT_PATH / "miotcd_samples.png"

plt.savefig(
    sample_output,
    dpi=150
)

print("=" * 60)
print("ĐÃ TẠO ẢNH MẪU")
print("=" * 60)

print(sample_output)

plt.show()