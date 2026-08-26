"""
Mục đích:
    Định nghĩa kiến trúc mô hình multi-head ResNet50.
    1 backbone dùng chung, 3 head phân loại riêng biệt.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


# =============================================================================
# ĐỊNH NGHĨA CLASSIFICATION HEAD
# =============================================================================

class ClassificationHead(nn.Module):

    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        # super().__init__() bắt buộc khi kế thừa nn.Module

        self.head = nn.Sequential(
            # Lớp 1: nén 2048 chiều → 512 chiều
            nn.Linear(in_features, 512),

            # BatchNorm1d: chuẩn hóa output của Linear
            # → ổn định quá trình training, hội tụ nhanh hơn
            nn.BatchNorm1d(512),

            # ReLU: hàm kích hoạt phi tuyến
            # → giúp model học các quan hệ phức tạp hơn
            nn.ReLU(inplace=True),

            # Dropout: ngẫu nhiên tắt dropout_rate% neuron trong lúc train
            # → buộc model không phụ thuộc vào 1 neuron cụ thể → giảm overfitting
            # inplace=False vì BatchNorm cần dữ liệu gốc
            nn.Dropout(p=dropout_rate),

            # Lớp 2: phân loại 512 chiều → num_classes
            nn.Linear(512, num_classes),
            # Không cần Softmax ở đây vì CrossEntropyLoss đã tích hợp sẵn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, 2048] — feature vector từ backbone
        Returns:
            logits (Tensor): [B, num_classes] — điểm số thô (chưa softmax)
        """
        return self.head(x)


# =============================================================================
# ĐỊNH NGHĨA MODEL CHÍNH
# =============================================================================

class VehicleMultiHeadModel(nn.Module):
    """
    Multi-head ResNet50 cho bài toán nhận diện thuộc tính phương tiện.

    Args:
        num_types  (int): số class TYPE  (mặc định 5)
        num_colors (int): số class COLOR (mặc định 15)
        num_makes  (int): số class MAKE  (mặc định 20)
        dropout_rate (float): dropout trong mỗi head
        freeze_backbone (bool): True = đóng băng backbone lúc train phase 1
    """

    def __init__(
        self,
        num_types:       int   = 5,
        num_colors:      int   = 15,
        num_makes:       int   = 20,
        dropout_rate:    float = 0.3,
        freeze_backbone: bool  = False,
    ):
        super().__init__()

        # -- Backbone: ResNet50 pretrained -------------------------------------
        # weights=ResNet50_Weights.DEFAULT → load pretrained ImageNet weights
        # Dùng pretrained giúp model đã biết nhận diện đặc trưng cơ bản
        # (cạnh, góc, texture,...) → train nhanh và tốt hơn
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # ResNet50 mặc định có lớp fc cuối: Linear(2048 → 1000) cho ImageNet
        # Ta cần bỏ lớp này đi, giữ lại phần feature extractor
        # backbone.fc.in_features = 2048
        in_features = backbone.fc.in_features  # = 2048

        # Thay lớp fc bằng Identity() = không làm gì cả
        # → output của backbone bây giờ là [B, 2048]
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # -- Đóng băng backbone nếu yêu cầu (Phase 1 training) ----------------
        if freeze_backbone:
            self.freeze_backbone()

        # -- 3 Classification Heads -------------------------------------------
        self.head_type  = ClassificationHead(in_features, num_types,  dropout_rate)
        self.head_color = ClassificationHead(in_features, num_colors, dropout_rate)
        self.head_make  = ClassificationHead(in_features, num_makes,  dropout_rate)

        print(f"  Model khởi tạo:")
        print(f"    Backbone  : ResNet50 (pretrained)")
        print(f"    Head TYPE : 2048 → 512 → {num_types}")
        print(f"    Head COLOR: 2048 → 512 → {num_colors}")
        print(f"    Head MAKE : 2048 → 512 → {num_makes}")
        print(f"    Dropout   : {dropout_rate}")
        print(f"    Backbone frozen: {freeze_backbone}")

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass qua toàn bộ model.

        Args:
            x (Tensor): [B, 3, 224, 224] — batch ảnh đã normalize

        Returns:
            (out_type, out_color, out_make): 3 tensor logits
                - out_type  [B, 5]
                - out_color [B, 15]
                - out_make  [B, 20]
        """

        # Bước 1: Trích xuất feature từ backbone
        features = self.backbone(x)
        # features shape: [B, 2048]
        # Mỗi ảnh được biểu diễn bằng 1 vector 2048 chiều

        # Bước 2: Đưa qua 3 head song song
        out_type  = self.head_type(features)   # [B, 5]
        out_color = self.head_color(features)  # [B, 15]
        out_make  = self.head_make(features)   # [B, 20]

        return out_type, out_color, out_make

    def freeze_backbone(self):
        """
        Đóng băng toàn bộ backbone — chỉ train 3 head.
        Dùng trong Phase 1 training để hội tụ nhanh.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        # requires_grad = False → không tính gradient → không update weight

    def unfreeze_backbone(self):
        """
        Mở băng backbone để train toàn bộ model.
        Dùng trong Phase 2 training để fine-tune.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def count_parameters(self) -> dict:
        """Đếm số lượng parameter có thể train và không thể train."""
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        frozen = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        return {
            "trainable": trainable,
            "frozen":    frozen,
            "total":     trainable + frozen,
        }


# =============================================================================
# HÀM TIỆN ÍCH
# =============================================================================

def build_model(
    num_types:       int   = 5,
    num_colors:      int   = 15,
    num_makes:       int   = 20,
    dropout_rate:    float = 0.3,
    freeze_backbone: bool  = True,   # Mặc định freeze cho Phase 1
    device: str = None,
) -> VehicleMultiHeadModel:
    """
    Tạo và trả về model đã sẵn sàng để train.

    Args:
        freeze_backbone (bool): True cho Phase 1, False cho Phase 2
        device (str): "cuda" / "cpu" / None (tự động detect)

    Returns:
        model (VehicleMultiHeadModel): model trên đúng device
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Thiết bị: {device}")

    model = VehicleMultiHeadModel(
        num_types=num_types,
        num_colors=num_colors,
        num_makes=num_makes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )

    model = model.to(device)
    # Chuyển toàn bộ weight lên GPU (nếu có)

    # In thống kê parameter
    params = model.count_parameters()
    print(f"\n  Parameters:")
    print(f"    Trainable : {params['trainable']:,}")
    print(f"    Frozen    : {params['frozen']:,}")
    print(f"    Total     : {params['total']:,}")

    return model


def load_checkpoint(checkpoint_path: Path, device: str = None) -> tuple:
    """
    Load model đã train từ file checkpoint .pth.

    Returns:
        (model, checkpoint): model đã load weight, và dict checkpoint đầy đủ
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Lấy số class từ checkpoint (lưu khi save)
    num_types  = checkpoint.get("num_types",  5)
    num_colors = checkpoint.get("num_colors", 15)
    num_makes  = checkpoint.get("num_makes",  20)

    model = VehicleMultiHeadModel(
        num_types=num_types,
        num_colors=num_colors,
        num_makes=num_makes,
        freeze_backbone=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Loaded checkpoint: {checkpoint_path.name}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')} | "
          f"Val Loss: {checkpoint.get('val_loss', '?'):.4f}")

    return model, checkpoint


# =============================================================================
# TEST NHANH
# =============================================================================

if __name__ == "__main__":
    """
    Chạy: python multi_head_model.py
    Kiểm tra model có khởi tạo và forward pass đúng không.
    """

    print("=" * 55)
    print("KIỂM TRA MULTI-HEAD MODEL")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Phase 1: Freeze backbone (chỉ train 3 head) --------------------------
    print("\n[Phase 1 - Backbone FROZEN]")
    model = build_model(freeze_backbone=True, device=device)

    # Test forward pass với batch giả
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    # torch.randn = tạo tensor ngẫu nhiên theo phân phối chuẩn
    # shape [4, 3, 224, 224] = 4 ảnh, 3 channel, 224x224

    model.eval()
    with torch.no_grad():
        out_type, out_color, out_make = model(dummy_input)

    print(f"\n  out_type  shape: {out_type.shape}")   # [4, 5]
    print(f"  out_color shape: {out_color.shape}")   # [4, 15]
    print(f"  out_make  shape: {out_make.shape}")    # [4, 20]

    # -- Phase 2: Unfreeze backbone (train toàn bộ) ---------------------------
    print("\n[Phase 2 - Backbone UNFROZEN]")
    model.unfreeze_backbone()
    params = model.count_parameters()
    print(f"  Trainable parameters: {params['trainable']:,}")

    print("\n  Model OK! Sẵn sàng để viết training script.")
