import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path


OLD_DATASET_DIR = Path(r"D:\TTTN_Co_Phuong\Dataset\vehicle_make\train")
VN_DATASET_DIR  = Path(r"D:\TTTN_Co_Phuong\Dataset\Make_Finetune_Data\sorted")
MERGED_DIR      = Path(r"D:\TTTN_Co_Phuong\Dataset\Make_Finetune_Data\merged")
OLD_MODEL_PATH  = Path(r"D:\TTTN_Co_Phuong\VehicleAttributeRecognition\VehicleAttributeRecognition\trained_models\best_vehicle_make_resnet50.pth")
NEW_MODEL_PATH  = Path(r"D:\TTTN_Co_Phuong\VehicleAttributeRecognition\VehicleAttributeRecognition\trained_models\best_vehicle_make_vn.pth")

# Chỉ train 12 hãng này (lấy từ danh sách user đã phân loại)
TARGET_CLASSES = sorted([
    'bmw', 'ford', 'honda', 'hyundai', 'jeep', 'kia',
    'mazda', 'mercedes', 'mitsubishi', 'suzuki', 'toyota', 'vinfast'
])

BATCH_SIZE = 16
EPOCHS     = 30
LR_HEAD    = 0.001  # LR cho lớp Classifier mới (train từ đầu)
LR_BODY    = 0.0001 # LR rất nhỏ cho phần Backbone đã đóng băng (fine-tune nhẹ)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Target classes ({len(TARGET_CLASSES)}): {TARGET_CLASSES}")

print("\n[1/4] Đang chuẩn bị Merged Dataset...")

if MERGED_DIR.exists():
    shutil.rmtree(MERGED_DIR)
MERGED_DIR.mkdir(parents=True)

for cls in TARGET_CLASSES:
    (MERGED_DIR / cls).mkdir()

img_count = {cls: 0 for cls in TARGET_CLASSES}

# Lấy ảnh từ Dataset cũ (chỉ lấy hãng có trong TARGET_CLASSES)
for cls in TARGET_CLASSES:
    src = OLD_DATASET_DIR / cls
    if src.exists():
        for f in src.iterdir():
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(f, MERGED_DIR / cls / f"old_{f.name}")
                img_count[cls] += 1

# Lấy ảnh từ Dataset VN mới (có Suzuki, VinFast)
for cls in TARGET_CLASSES:
    src = VN_DATASET_DIR / cls
    if src.exists():
        for f in src.iterdir():
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(f, MERGED_DIR / cls / f"vn_{f.name}")
                img_count[cls] += 1

print("Số ảnh mỗi hãng sau khi trộn:")
for cls, cnt in img_count.items():
    print(f"  {cls:15s}: {cnt} ảnh")
print(f"  Tổng cộng: {sum(img_count.values())} ảnh")


print("\n[2/4] Đang chuẩn bị DataLoader...")

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset   = datasets.ImageFolder(str(MERGED_DIR), transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

class_names = dataset.classes
num_classes = len(class_names)
print(f"ImageFolder đọc được {len(dataset)} ảnh, {num_classes} class: {class_names}")


print("\n[3/4] Đang load model và thiết lập Freeze Layers...")

# Load Feature Extractor từ model cũ (20 classes)
checkpoint = torch.load(OLD_MODEL_PATH, map_location=DEVICE)
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 20)
model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

# Đóng băng toàn bộ Backbone (layer1-4), chỉ để mở lớp fc
for name, param in model.named_parameters():
    if name.startswith("fc"):
        param.requires_grad = True  # Lớp đầu ra: train bình thường
    else:
        param.requires_grad = True  # Cho phép fine-tune nhẹ backbone

# Thay lớp đầu ra với 12 classes mới
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# Dùng 2 Learning Rate khác nhau: backbone nhỏ, head lớn hơn
backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
head_params     = [p for n, p in model.named_parameters() if n.startswith("fc")]

optimizer = optim.Adam([
    {"params": backbone_params, "lr": LR_BODY},
    {"params": head_params,     "lr": LR_HEAD}
])
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion  = nn.CrossEntropyLoss()


print("\n[4/4] Bắt đầu Training...")
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

    scheduler.step()
    epoch_loss = running_loss / total
    epoch_acc  = correct / total * 100

    marker = " ★ BEST" if epoch_acc > best_acc else ""
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), NEW_MODEL_PATH)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%{marker}")

print(f"\n[XONG] Best Accuracy: {best_acc:.2f}%")
print(f"[XONG] Model đã lưu tại: {NEW_MODEL_PATH}")
print(f"[XONG] Class order: {class_names}")
