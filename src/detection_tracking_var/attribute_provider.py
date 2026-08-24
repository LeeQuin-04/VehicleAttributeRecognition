#attribute_provider.py
from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet50


class ResNet50Classifier:

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
    ) -> None:

        self.device = device

        # Load checkpoint
        checkpoint = torch.load(
            model_path,
            map_location=device,
        )

        self.class_names = checkpoint["class_names"]

        num_classes = len(self.class_names)

        self.model = resnet50(
            weights=None
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(
            f"Loaded: {model_path.name}"
        )

        print(
            f"Classes ({num_classes}): "
            f"{self.class_names}"
        )


    def predict(
        self,
        image,
    ) -> tuple[str, float]:

        if image is None or image.size == 0:
            return "unknown", 0.0

        # OpenCV: BGR
        # Dataset/ImageFolder/PIL: RGB
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image_tensor = self.transform(
            image_rgb
        )

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            outputs = self.model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        class_name = self.class_names[
            prediction.item()
        ]

        return (
            class_name,
            float(confidence.item())
        )


class VehicleAttributeProvider:

    def __init__(self) -> None:

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        models_dir = (
            project_root
            / "models"
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Attribute models device: "
            f"{self.device}"
        )

        # Model nhận diện màu
        self.color_model = ResNet50Classifier(
            model_path=(
                models_dir
                / "color"
                / "best_vehicle_color_resnet50.pth"
            ),
            device=self.device
        )

        # Model nhận diện loại xe
        self.type_model = ResNet50Classifier(
            model_path=(
                models_dir
                / "type"
                / "best_vehicle_type_resnet50.pth"
            ),
            device=self.device
        )

        # Model nhận diện hãng xe
        self.make_model = ResNet50Classifier(
            model_path=(
                models_dir
                / "make"
                / "best_vehicle_make_resnet50.pth"
            ),
            device=self.device
        )


    def predict(
        self,
        vehicle_image,
    ) -> dict:

        color, color_confidence = (
            self.color_model.predict(
                vehicle_image
            )
        )

        vehicle_type, type_confidence = (
            self.type_model.predict(
                vehicle_image
            )
        )

        make, make_confidence = (
            self.make_model.predict(
                vehicle_image
            )
        )

        return {
            "color": color,
            "color_confidence": color_confidence,

            "vehicle_type": vehicle_type,
            "type_confidence": type_confidence,

            "make": make,
            "make_confidence": make_confidence,
        }