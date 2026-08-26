"""
Kien truc Hybrid:
    - Multi-Head Model  => TYPE + COLOR
    - Make Model rieng  => MAKE (chinh xac hon cho hang xe)

Thay doi so voi phien ban truoc:
    - TAT CA loai xe (ca bus/truck/van) deu duoc predict MAKE
    - Neu confidence < MAKE_CONFIDENCE_THRESHOLD (30%) => hien "Unknown"
    - Khong con logic "N/A" cung cac loai xe

Quy trinh:
    Anh => Multi-Head => TYPE + COLOR
                     => Make Model => MAKE (ap dung cho moi loai xe)
                                   => "Unknown" neu confidence < 30%
"""

import json
from pathlib import Path
from typing import Union, Dict, Any

import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model.multi_head_model import load_checkpoint


# =============================================================================
# CAU HINH
# =============================================================================

DEFAULT_MULTIHEAD_PATH = PROJECT_ROOT / "trained_models" / "best_multi_head.pth"
DEFAULT_MAKE_PATH      = PROJECT_ROOT / "trained_models" / "best_vehicle_make_resnet50.pth"
DEFAULT_CLASS_INFO     = PROJECT_ROOT / "data" / "unified" / "class_info.json"

# Thu tu nay phai khop voi luc train model make cu
MAKE_CLASSES = [
    "audi", "bmw", "cadillac", "chevrolet", "dodge",
    "ford", "gmc", "honda", "hyundai", "infiniti",
    "jeep", "kia", "landrover", "lexus", "mazda",
    "mercedes", "mitsubishi", "nissan", "porsche", "toyota",
]

# Nguong tin cay toi thieu de hien thi ket qua make
# Neu confidence < nguong nay => hien "Unknown" thay vi doan bua
# 30% — ap dung cho TAT CA loai xe (ca bus/truck/van)
MAKE_CONFIDENCE_THRESHOLD = 0.30


# =============================================================================
# HAM LOAD MAKE MODEL (ResNet50 don le)
# =============================================================================

def load_make_model(model_path: Path, device: str) -> torch.nn.Module:
    """Load model make rieng le best_vehicle_make_resnet50.pth."""
    checkpoint = torch.load(model_path, map_location=device)

    num_classes = len(MAKE_CLASSES)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Doc dung key tu checkpoint
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    print(f"  Make model loaded: {model_path.name} ({num_classes} classes)")
    return model


# =============================================================================
# LOP PREDICTOR HYBRID
# =============================================================================

class VehiclePredictor:
    def __init__(
        self,
        multihead_path:  Union[str, Path] = None,
        make_model_path: Union[str, Path] = None,
        class_info_path: Union[str, Path] = None,
        device:          str   = None,
        make_threshold:  float = MAKE_CONFIDENCE_THRESHOLD,
    ):
        self.multihead_path  = Path(multihead_path  or DEFAULT_MULTIHEAD_PATH)
        self.make_model_path = Path(make_model_path or DEFAULT_MAKE_PATH)
        self.class_info_path = Path(class_info_path or DEFAULT_CLASS_INFO)
        self.make_threshold  = make_threshold

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Khoi tao Hybrid Predictor | Device: {self.device}")

        self._load_class_info()
        self._load_multihead_model()
        self._load_make_model()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load_class_info(self):
        with open(self.class_info_path, "r", encoding="utf-8") as f:
            class_info = json.load(f)
        self.idx_to_class = {}
        for task in ["type", "color"]:
            self.idx_to_class[task] = {int(v): k for k, v in class_info[task].items()}
        # Make dung MAKE_CLASSES truc tiep tu model cu
        self.idx_to_make = {i: c for i, c in enumerate(MAKE_CLASSES)}

    def _load_multihead_model(self):
        self.multihead_model, _ = load_checkpoint(self.multihead_path, device=self.device)
        self.multihead_model.eval()

    def _load_make_model(self):
        self.make_model = load_make_model(self.make_model_path, self.device)

    def _preprocess(self, image_input) -> torch.Tensor:
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Loai khong ho tro: {type(image_input)}")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_input) -> Dict[str, Any]:
        """
        Du doan TYPE + COLOR + MAKE cho 1 anh.
        TAT CA loai xe (ca bus/truck/van) deu duoc predict MAKE.
        Neu confidence < make_threshold => "Unknown".
        Returns:
            {"type": {"class":..., "confidence":...},
             "color": ..., "make": ...}
        """
        tensor = self._preprocess(image_input)

        # Buoc 1: Multi-Head => TYPE + COLOR (bo qua make head)
        out_type, out_color, _ = self.multihead_model(tensor)
        prob_type  = F.softmax(out_type,  dim=1).squeeze(0)
        prob_color = F.softmax(out_color, dim=1).squeeze(0)

        idx_type  = prob_type.argmax().item()
        idx_color = prob_color.argmax().item()

        pred_type  = self.idx_to_class["type"].get(idx_type,  "unknown")
        pred_color = self.idx_to_class["color"].get(idx_color, "unknown")
        conf_type  = float(prob_type[idx_type])
        conf_color = float(prob_color[idx_color])

        # Buoc 2: Make Model chuyen biet => MAKE (ap dung cho TAT CA loai xe)
        out_make  = self.make_model(tensor)
        prob_make = F.softmax(out_make, dim=1).squeeze(0)
        idx_make  = prob_make.argmax().item()
        conf_make = float(prob_make[idx_make])

        if conf_make >= self.make_threshold:
            pred_make = self.idx_to_make.get(idx_make, "unknown")
        else:
            pred_make = "Unknown"  # Confidence qua thap, khong doan bua

        return {
            "type":  {"class": pred_type,  "confidence": conf_type},
            "color": {"class": pred_color, "confidence": conf_color},
            "make":  {"class": pred_make,  "confidence": conf_make},
        }


# =============================================================================
# CHAY TEST TU DONG LENH
# =============================================================================

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Hybrid Vehicle Attribute Predictor")
    parser.add_argument("image_path", type=str, nargs="?")
    parser.add_argument("--threshold", type=float, default=MAKE_CONFIDENCE_THRESHOLD,
                        help=f"Nguong confidence make (mac dinh: {MAKE_CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()

    if not args.image_path:
        print("Cach dung: python predictor.py <duong_dan_anh> [--threshold 0.4]")
        sys.exit(1)

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"[LOI] Khong tim thay anh: {img_path}")
        sys.exit(1)

    print("=" * 60)
    print("HYBRID VEHICLE PREDICTOR")
    print("=" * 60)

    predictor = VehiclePredictor(make_threshold=args.threshold)

    print(f"\nAnh: {img_path.name}")
    result = predictor.predict(img_path)

    print("\nKET QUA DU DOAN:")
    print(f"  Loai xe : {result['type']['class']:10}  ({result['type']['confidence']:.2%})")
    print(f"  Mau sac : {result['color']['class']:10}  ({result['color']['confidence']:.2%})")
    print(f"  Hang xe : {result['make']['class']:10}  ({result['make']['confidence']:.2%})")

    if result["make"]["class"] == "Unknown":
        print(f"\n  [Ghi chu] Confidence make qua thap => hien 'Unknown'")
        print(f"  Thu giam nguong: python predictor.py {img_path} --threshold 0.25")
