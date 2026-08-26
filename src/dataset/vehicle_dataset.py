"""
Mục đích:
    PyTorch Dataset class — đọc file CSV (train/val/test),
    load từng ảnh, và trả về tensor + nhãn số để dùng trong training.
"""

import json
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =============================================================================
# CẤU HÌNH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Đường dẫn đến 3 file CSV và class_info.json
UNIFIED_DIR  = PROJECT_ROOT / "data" / "unified"
TRAIN_CSV    = UNIFIED_DIR / "train.csv"
VAL_CSV      = UNIFIED_DIR / "val.csv"
TEST_CSV     = UNIFIED_DIR / "test.csv"
CLASS_INFO   = UNIFIED_DIR / "class_info.json"

# Kích thước ảnh đầu vào của ResNet50
IMAGE_SIZE = 224

# Batch size khi train — giảm xuống nếu GPU bị out of memory
BATCH_SIZE = 32

# Số worker để load ảnh song song
# Trên Windows nên để 0 (tránh lỗi multiprocessing)
# Trên Linux/Colab có thể để 4
NUM_WORKERS = 0


# =============================================================================
# ĐỊNH NGHĨA TRANSFORM ẢNH
# =============================================================================

# Transform cho tập TRAIN — có augmentation (tăng cường dữ liệu)
# Augmentation giúp model học được các trường hợp khác nhau
# (ảnh bị lật, màu sắc khác nhau, góc độ khác nhau,...)
TRAIN_TRANSFORM = transforms.Compose([
    # Resize cạnh ngắn về 256
    transforms.Resize(256),

    # ─── AUGMENTATION HÌNH HỌC ───
    # Random Crop 224x224
    transforms.RandomCrop(IMAGE_SIZE),

    # Lật ngang 50%
    transforms.RandomHorizontalFlip(p=0.5),

    # Xoay ngẫu nhiên ±15 độ
    transforms.RandomRotation(degrees=15),

    # Perspective (phối cảnh 3D) — mô phỏng ảnh chụp từ góc độ khác nhau
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),

    # Affine transform nhẹ (dịch chuyển + co dãn) — mô phỏng xe to nhỏ, lệch khung hình
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),

    # ─── AUGMENTATION MÀU SẮC ───
    # Thay đổi sáng, tương phản, màu mạnh hơn trước
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.08,
    ),

    # Chuyển ảnh thành grayscale 15% thời gian — giúp model ít phụ thuộc màu quá
    transforms.RandomGrayscale(p=0.15),

    # ─── CHUẨN HÓA ───
    # Chuyển PIL Image → Tensor [0.0, 1.0]
    transforms.ToTensor(),

    # Xóa ngẫu nhiên 1 vùng nhỏ trong ảnh (mô phỏng bị che khuất)
    # Phải đặt SAU ToTensor() vì RandomErasing hoạt động trên tensor
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3)),

    # Normalize theo ImageNet (bắt buộc vì dùng ResNet50 pretrained)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Transform cho VAL và TEST — KHÔNG có augmentation
# Mục đích: đánh giá nhất quán, không ngẫu nhiên
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),   # Crop chính giữa (cố định)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =============================================================================
# DATASET CLASS
# =============================================================================

class VehicleDataset(Dataset):
    """
    Dataset cho bài toán nhận diện thuộc tính phương tiện.

    Kế thừa từ torch.utils.data.Dataset — bắt buộc phải implement:
        __len__():      trả về tổng số mẫu
        __getitem__():  trả về 1 mẫu theo index

    Args:
        csv_path    (Path): đường dẫn đến file CSV (train/val/test)
        class_info  (dict): mapping tên class → index (từ class_info.json)
        transform   (callable): transform áp dụng lên ảnh
    """

    def __init__(self, csv_path: Path, class_info: dict, transform=None):

        # Đọc CSV vào DataFrame
        self.df = pd.read_csv(csv_path)
        # self.df có dạng:
        # image_path | source | type  | color | make
        # /path/a.jpg| miotcd | bus   | black | N/A
        # /path/b.jpg| ...    | car   | red   | toyota

        self.class_info = class_info
        # class_info["type"]  = {"bus":0, "car":1, "pickup":2, "truck":3, "van":4}
        # class_info["color"] = {"beige":0, "black":1, ..., "yellow":14}
        # class_info["make"]  = {"audi":0, ..., "toyota":18, "N/A":-1}

        self.transform = transform

        print(f"  Dataset loaded: {csv_path.name} | {len(self.df)} ảnh")

    def __len__(self) -> int:
        """Trả về tổng số dòng trong CSV = tổng số ảnh."""
        return len(self.df)

    def __getitem__(self, index: int) -> tuple:
        """
        Trả về 1 mẫu theo index.

        Args:
            index (int): vị trí dòng trong CSV (0 đến len-1)

        Returns:
            tuple: (image, type_idx, color_idx, make_idx)
                - image     (Tensor): [3, 224, 224]
                - type_idx  (int)   : 0..4
                - color_idx (int)   : 0..14
                - make_idx  (int)   : 0..18 hoặc -1 nếu N/A
        """

        # Lấy 1 dòng từ DataFrame theo index
        row = self.df.iloc[index]
        # row["image_path"] = "D:/.../.../car.jpg"
        # row["type"]       = "car"
        # row["color"]      = "black"
        # row["make"]       = "toyota"

        # -- Bước 1: Load ảnh -------------------------------------------------
        image_path = Path(row["image_path"])

        try:
            image = Image.open(image_path).convert("RGB")
            # convert("RGB") đảm bảo ảnh luôn có 3 channel
        except Exception:
            # Nếu ảnh bị lỗi → tạo ảnh đen thay thế để không crash training
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)

        # -- Bước 2: Áp dụng transform -----------------------------------------
        if self.transform is not None:
            image = self.transform(image)
        # Sau bước này: image là Tensor shape [3, 224, 224]

        # -- Bước 3: Chuyển nhãn text → index số --------------------------------
        type_idx  = self.class_info["type"].get(row["type"],  -1)
        color_idx = self.class_info["color"].get(row["color"], -1)
        make_idx  = self.class_info["make"].get(row["make"],   -1)
        # .get(key, default=-1): nếu không tìm thấy key → trả -1
        # make_idx = -1 khi make = "N/A" (bus/truck/van)

        return image, type_idx, color_idx, make_idx


# =============================================================================
# HÀM TẠO DATALOADER
# =============================================================================

def get_dataloaders(
    batch_size:  int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple:
    """
    Tạo 3 DataLoader cho train / val / test.

    DataLoader = wrapper xung quanh Dataset:
        - Tự động chia batch (nhóm nhiều ảnh lại)
        - Shuffle (xáo trộn) khi train
        - Load song song bằng num_workers

    Returns:
        (train_loader, val_loader, test_loader)
    """

    # -- Đọc class_info.json ---------------------------------------------------
    with open(CLASS_INFO, "r", encoding="utf-8") as f:
        class_info = json.load(f)
    # class_info["make"]["N/A"] = -1  ← đã lưu trong file này

    print("=" * 50)
    print("KHỞI TẠO DATASET")
    print("=" * 50)

    # -- Tạo 3 Dataset ---------------------------------------------------------
    train_dataset = VehicleDataset(TRAIN_CSV, class_info, transform=TRAIN_TRANSFORM)
    val_dataset   = VehicleDataset(VAL_CSV,   class_info, transform=VAL_TRANSFORM)
    test_dataset  = VehicleDataset(TEST_CSV,  class_info, transform=VAL_TRANSFORM)
    # Val và Test dùng VAL_TRANSFORM (không augmentation)

    # -- Tạo 3 DataLoader ------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Xáo trộn khi train để model không học thứ tự
        num_workers=num_workers,
        pin_memory=True,       # Tăng tốc transfer CPU→GPU (chỉ hiệu quả khi có CUDA)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # Không xáo trộn khi đánh giá
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n  Train: {len(train_dataset)} ảnh | {len(train_loader)} batches")
    print(f"  Val  : {len(val_dataset)}  ảnh | {len(val_loader)} batches")
    print(f"  Test : {len(test_dataset)} ảnh | {len(test_loader)} batches")
    print(f"  Batch size : {batch_size}")

    return train_loader, val_loader, test_loader


# =============================================================================
# TEST NHANH (chạy trực tiếp để kiểm tra)
# =============================================================================

if __name__ == "__main__":
    """
    Chạy để kiểm tra dataset có hoạt động không:
        python vehicle_dataset.py

    Nếu không có lỗi và in ra được shape của batch → OK
    """

    print("=" * 50)
    print("KIỂM TRA VEHICLE DATASET")
    print("=" * 50)

    # Tạo DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)

    # Lấy 1 batch đầu tiên từ train_loader để kiểm tra
    print("\n  Lấy 1 batch mẫu từ train_loader...")
    images, type_idxs, color_idxs, make_idxs = next(iter(train_loader))
    # next(iter(...)) = lấy phần tử đầu tiên của iterator

    print(f"\n  images shape     : {images.shape}")
    # Kỳ vọng: torch.Size([4, 3, 224, 224])
    # [batch_size=4, channels=3, height=224, width=224]

    print(f"  type_idxs        : {type_idxs.tolist()}")
    # Vd: [0, 2, 1, 3]  (bus, pickup, car, truck)

    print(f"  color_idxs       : {color_idxs.tolist()}")
    # Vd: [1, 5, 13, 6]  (black, green, white, grey)

    print(f"  make_idxs        : {make_idxs.tolist()}")
    # Vd: [16, -1, 18, -1]  (nissan, N/A, toyota, N/A)
    # -1 = bus/truck/van, sẽ bị bỏ qua khi tính loss make

    print("\n  Dataset OK! Sẵn sàng để train.")
    print("  Bước tiếp theo: Viết multi_head_model.py")
