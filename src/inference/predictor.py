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


DEFAULT_COLOR_PATH = PROJECT_ROOT / "trained_models" / "best_vehicle_color_resnet50.pth"
DEFAULT_MAKE_PATH  = PROJECT_ROOT / "trained_models" / "best_vehicle_make_vn.pth"
DEFAULT_CLASS_INFO = PROJECT_ROOT / "data" / "unified" / "class_info.json"

# Danh sach 12 hang xe theo dataset VN moi
MAKE_CLASSES = [
    'bmw', 'ford', 'honda', 'hyundai', 'jeep', 'kia',
    'mazda', 'mercedes', 'mitsubishi', 'suzuki', 'toyota', 'vinfast'
]

MAKE_CONFIDENCE_THRESHOLD = 0.30


def load_resnet50_model(model_path: Path, num_classes: int, device: str) -> torch.nn.Module:
    """Load model ResNet50 tu checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    
    print(f"  [+] Model loaded: {model_path.name} ({num_classes} classes)")
    return model


class VehiclePredictor:
    def __init__(
        self,
        color_model_path: Union[str, Path] = None,
        make_model_path:  Union[str, Path] = None,
        class_info_path:  Union[str, Path] = None,
        device:           str   = None,
        make_threshold:   float = MAKE_CONFIDENCE_THRESHOLD,
    ):
        self.color_model_path = Path(color_model_path or DEFAULT_COLOR_PATH)
        self.make_model_path  = Path(make_model_path or DEFAULT_MAKE_PATH)
        self.class_info_path  = Path(class_info_path or DEFAULT_CLASS_INFO)
        self.make_threshold   = make_threshold

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Khoi tao VehiclePredictor | Device: {self.device}")

        self._load_class_info()
        
        # Load Color Model
        num_colors = len(self.idx_to_class["color"])
        self.color_model = load_resnet50_model(self.color_model_path, num_colors, self.device)
        
        # Load Make Model
        num_makes = len(self.idx_to_make)
        self.make_model = load_resnet50_model(self.make_model_path, num_makes, self.device)

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
            if task in class_info:
                self.idx_to_class[task] = {int(v): k for k, v in class_info[task].items()}
        self.idx_to_make = {i: c for i, c in enumerate(MAKE_CLASSES)}

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
    def predict(self, image_input, yolo_type: str = None) -> Dict[str, Any]:
        """
        Du doan COLOR + MAKE. (Type duoc truyen tu YOLO vao)
        CHI doan MAKE cho car.
        """
        tensor = self._preprocess(image_input)

        # 1. Color Model
        out_color = self.color_model(tensor)
        prob_color = F.softmax(out_color, dim=1).squeeze(0)
        idx_color = prob_color.argmax().item()
        pred_color = self.idx_to_class["color"].get(idx_color, "unknown")
        conf_color = float(prob_color[idx_color])
        
        # 2. Type (Lay truc tiep tu tham so YOLO)
        pred_type = yolo_type if yolo_type else "unknown"
        conf_type = 1.0

        # 3. Make Model (CHI CAR)
        if pred_type.lower() == "car":
            out_make  = self.make_model(tensor)
            prob_make = F.softmax(out_make, dim=1).squeeze(0)
            idx_make  = prob_make.argmax().item()
            conf_make = float(prob_make[idx_make])
            
            if conf_make >= self.make_threshold:
                pred_make = self.idx_to_make.get(idx_make, "unknown")
            else:
                pred_make = "Unknown"
        else:
            pred_make = "N/A"
            conf_make = 1.0

        return {
            "type":  {"class": pred_type,  "confidence": conf_type},
            "color": {"class": pred_color, "confidence": conf_color},
            "make":  {"class": pred_make,  "confidence": conf_make},
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()
    
    predictor = VehiclePredictor()
    res = predictor.predict(args.image_path, yolo_type="car")
    print("\nKet qua (Gia lap yolo_type='car'):")
    print(json.dumps(res, indent=4))
