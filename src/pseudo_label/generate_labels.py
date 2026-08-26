"""
Mục đích:
    Ba dataset gốc mỗi bộ chỉ có 1 loại nhãn:
      - MIO-TCD         → chỉ có nhãn TYPE  (car/bus/van/...)
      - Vehicle Color   → chỉ có nhãn COLOR (black/white/...)
      - Vehicle Make    → chỉ có nhãn MAKE  (toyota/honda/...)

    Script này dùng 3 model ResNet50 đã train sẵn để "đoán" (pseudo-label)
    các nhãn còn thiếu, rồi ghi tất cả ra 3 file CSV riêng.

Kết quả: mỗi CSV có các cột:
    image_path | source | type | color | make
               |        |      |       |
               |        |  <- nhãn THẬT nếu dataset đó có
               |        |  <- nhãn GIẢ  nếu phải dùng model đoán

"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image


# =============================================================================
# CẤU HÌNH - ĐƯỜNG DẪN & THÔNG SỐ
# =============================================================================

# Thư mục gốc của project
# Path(__file__) = đường dẫn đến file script này
# .parent = thư mục chứa file này (src/pseudo_label/)
# .parent.parent = src/
# .parent.parent.parent = VehicleAttributeRecognition/ (thư mục project)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# -- Đường dẫn tới 3 model đã train ------------------------------------------
MODEL_DIR = PROJECT_ROOT.parent / "trained_models"
#   PROJECT_ROOT = ...VehicleAttributeRecognition/VehicleAttributeRecognition/
#   .parent      = ...VehicleAttributeRecognition/
#   / "trained_models" = ...VehicleAttributeRecognition/trained_models/

MODEL_TYPE_PATH  = MODEL_DIR / "best_vehicle_type_resnet50.pth"
MODEL_COLOR_PATH = MODEL_DIR / "best_vehicle_color_resnet50.pth"
MODEL_MAKE_PATH  = MODEL_DIR / "best_vehicle_make_resnet50.pth"

# -- Đường dẫn tới 3 dataset gốc ----------------------------------------------
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# MIO-TCD: ảnh nằm trong data/raw/miotcd/train1/<class_name>/<image.jpg>
MIOTCD_PATH = DATA_RAW / "miotcd" / "train"

# Vehicle Color: ảnh nằm trong data/raw/vehicle_color/train/<class_name>/<image.jpg>
COLOR_DATASET_PATH = DATA_RAW / "vehicle_color"

# Vehicle Make: ảnh nằm trong data/raw/vehicle_make/<make_name>/.../<image.jpg>
#   Chú ý: dataset này có cấu trúc lồng nhau nhiều cấp, dùng rglob để tìm ảnh
MAKE_DATASET_PATH = DATA_RAW / "vehicle_make" / "train"

# !! Sửa lại các đường dẫn trên nếu dataset của bạn ở chỗ khác !!

# -- Đường dẫn lưu file CSV output --------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "data" / "pseudo_labeled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# parents=True  → tạo cả thư mục cha nếu chưa có
# exist_ok=True → không báo lỗi nếu thư mục đã tồn tại

CSV_MIOTCD = OUTPUT_DIR / "miotcd_labeled.csv"
CSV_COLOR  = OUTPUT_DIR / "color_labeled.csv"
CSV_MAKE   = OUTPUT_DIR / "make_labeled.csv"

# -- Danh sách class cho từng task --------------------------------------------

# TYPE: 5 classes (đã được map từ MIO-TCD gốc trong prepare_miotcd.py)
TYPE_CLASSES = ["bus", "car", "pickup", "truck", "van"]
# Lưu ý: sắp xếp alphabet vì thường lúc train ImageFolder tự sort tên folder

# COLOR: 15 classes
COLOR_CLASSES = [
    "beige", "black", "blue", "brown", "gold",
    "green", "grey", "orange", "pink", "purple",
    "red", "silver", "tan", "white", "yellow",
]
# Index: 0=beige, 1=black, 2=blue, ..., 14=yellow

# MAKE: 20 hãng xe
MAKE_CLASSES = [
    "audi", "bmw", "cadillac", "chevrolet", "dodge",
    "ford", "gmc", "honda", "hyundai", "infiniti",
    "jeep", "kia", "landrover", "lexus", "mazda",
    "mercedes", "mitsubishi", "nissan", "porsche", "toyota",
]

# !! QUAN TRỌNG: Thứ tự danh sách phải KHỚP với thứ tự class lúc train model !!
# Nếu lúc train dùng ImageFolder (PyTorch) thì nó tự sort alphabet → danh sách trên đã OK
# Nếu không chắc → kiểm tra lại bằng cách in model output và so với ảnh mẫu

# -- Ngưỡng confidence --------------------------------------------------------
# Nếu model dự đoán với xác suất < CONFIDENCE_THRESHOLD → ghi "unknown"
# Tránh đưa vào training những nhãn mà model không chắc chắn
CONFIDENCE_THRESHOLD = 0.50   # 50% — có thể điều chỉnh tùy kết quả

# -- Định dạng ảnh được chấp nhận ---------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# -- Thiết bị tính toán -------------------------------------------------------
# Nếu máy có GPU (CUDA) → dùng GPU, không thì dùng CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {DEVICE}")


# =============================================================================
# TRANSFORM ẢNH
# =============================================================================

# Transform này PHẢI GIỐNG với transform dùng lúc train model
# Vì model được train với ảnh 224x224 và normalize theo ImageNet
IMAGE_TRANSFORM = transforms.Compose([
    # Resize ảnh về kích thước 256 (cạnh ngắn hơn = 256, giữ tỷ lệ)
    transforms.Resize(256),

    # Crop chính giữa lấy vùng 224x224
    # → kết quả cuối là ảnh 224x224 (kích thước input chuẩn của ResNet50)
    transforms.CenterCrop(224),

    # Chuyển PIL Image sang PyTorch Tensor
    # Đồng thời scale pixel từ [0, 255] → [0.0, 1.0]
    transforms.ToTensor(),

    # Normalize theo giá trị mean và std của ImageNet
    # Công thức: (pixel - mean) / std  (áp dụng cho từng channel RGB)
    # Lý do: ResNet50 pretrain trên ImageNet với chuẩn hóa này, giữ nguyên thì model hoạt động tốt hơn
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =============================================================================
# PHẦN 4: HÀM LOAD MODEL
# =============================================================================

def load_model(model_path: Path, num_classes: int) -> nn.Module:
    """
    Load 1 model ResNet50 từ file .pth.

    Tham số:
        model_path  (Path): đường dẫn đến file .pth
        num_classes (int) : số class mà model này phân loại

    Trả về:
        model (nn.Module): model đã load weight, ở chế độ eval, sẵn sàng inference
    """

    # Bước 1: Khởi tạo kiến trúc ResNet50
    # weights=None vì chúng ta sẽ load weight riêng từ file .pth
    # Không cần load pretrained ImageNet weights ở đây
    model = models.resnet50(weights=None)

    # Bước 2: Thay thế lớp fc (fully connected) cuối cùng
    # ResNet50 mặc định: fc = Linear(2048 → 1000) cho ImageNet 1000 class
    # Ta cần:          fc = Linear(2048 → num_classes) cho bài toán của mình
    #
    # model.fc.in_features = 2048 (là kích thước feature vector trước lớp fc)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Sau dòng này, kiến trúc model giống hệt lúc được train → mới load được weight

    # Bước 3: Load weight từ file .pth
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    # In class_names lưu trong checkpoint để xác nhận thứ tự class
    if "class_names" in checkpoint:
        print(f"    class_names trong file: {checkpoint['class_names']}")

    # Bước 4: Chuyển model sang thiết bị tính toán
    model = model.to(DEVICE)

    # Bước 5: Chuyển sang chế độ eval (evaluation mode)
    model.eval()
    # eval() có 2 tác dụng quan trọng:
    #   - Tắt Dropout (Dropout chỉ dùng lúc training để tránh overfitting)
    #   - BatchNorm dùng running statistics thay vì batch statistics
    # LUÔN gọi .eval() trước khi inference!

    print(f"  → Đã load: {model_path.name}  ({num_classes} classes)")
    return model


# =============================================================================
# PHẦN 5: HÀM INFERENCE 1 ẢNH
# =============================================================================

def predict(model: nn.Module, image_path: Path, class_names: list) -> tuple:
    """
    Dùng model để dự đoán class của 1 ảnh.

    Tham số:
        model       (nn.Module): model đã load
        image_path  (Path)     : đường dẫn đến ảnh
        class_names (list)     : danh sách tên class (vd: ["car","bus","van",...])

    Trả về:
        (predicted_class, confidence)
        - predicted_class (str)  : tên class dự đoán, hoặc "unknown" nếu confidence thấp
        - confidence      (float): xác suất của class được chọn (0.0 đến 1.0)
    """

    # Bước 1: Mở ảnh
    try:
        img = Image.open(image_path)
        # Chuyển sang RGB để đảm bảo ảnh có đúng 3 channel
        # Một số ảnh PNG có thể là RGBA (4 channel), grayscale (1 channel)
        # → convert("RGB") xử lý hết các trường hợp này
        img = img.convert("RGB")
    except Exception as e:
        # Ảnh bị hỏng hoặc sai format → trả về "unknown"
        print(f"    Không đọc được ảnh: {image_path.name} | Lỗi: {e}")
        return "unknown", 0.0

    # Bước 2: Áp dụng transform
    # IMAGE_TRANSFORM trả về tensor shape [3, 224, 224]
    #   3   = số channel (RGB)
    #   224 = chiều cao (pixel)
    #   224 = chiều rộng (pixel)
    tensor = IMAGE_TRANSFORM(img)

    # Model luôn nhận input dạng batch: [batch_size, 3, 224, 224]
    # Vì ta chỉ có 1 ảnh, cần thêm dimension batch ở đầu:
    # [3, 224, 224] → [1, 3, 224, 224]
    # unsqueeze(0) = thêm 1 dimension tại vị trí 0
    tensor = tensor.unsqueeze(0)

    # Chuyển tensor lên thiết bị tính toán (GPU hoặc CPU)
    tensor = tensor.to(DEVICE)

    # Bước 3: Forward pass (inference)
    # torch.no_grad(): tắt tính gradient
    #   → tiết kiệm bộ nhớ và tăng tốc vì inference không cần backpropagation
    with torch.no_grad():
        # output shape: [1, num_classes]
        # Đây là "logits" = điểm số thô, chưa normalize, có thể âm
        output = model(tensor)

    # Bước 4: Tính xác suất bằng Softmax
    # Softmax chuyển logits thành xác suất:
    #   - Tất cả giá trị nằm trong [0, 1]
    #   - Tổng tất cả class = 1.0
    # dim=1: tính softmax theo chiều class (chiều số 1)
    probabilities = F.softmax(output, dim=1)
    # probabilities shape: [1, num_classes]

    # Bước 5: Lấy class có xác suất cao nhất
    # torch.max() trả về (tensor giá trị lớn nhất, tensor index)
    # dim=1: tìm max theo chiều class
    # .item() chuyển tensor 1 phần tử → số Python thông thường (float/int)
    confidence, predicted_idx = torch.max(probabilities, dim=1)
    confidence    = confidence.item()      # float, vd: 0.87
    predicted_idx = predicted_idx.item()   # int,   vd: 2

    # Bước 6: Kiểm tra ngưỡng confidence
    if confidence < CONFIDENCE_THRESHOLD:
        # Model không tự tin → không dùng nhãn giả này
        # Ảnh có nhãn "unknown" sẽ bị lọc bỏ ở bước build_unified_dataset
        return "unknown", confidence

    # Lấy tên class từ danh sách
    predicted_class = class_names[predicted_idx]
    return predicted_class, confidence


# =============================================================================
# PHẦN 6: XỬ LÝ TỪNG DATASET
# =============================================================================

def process_miotcd(model_color: nn.Module, model_make: nn.Module) -> None:
    """
    Xử lý MIO-TCD:
        - TYPE : lấy từ tên folder (nhãn THẬT)
        - COLOR: dùng model_color dự đoán (PSEUDO)
        - MAKE : dùng model_make  dự đoán (PSEUDO)

    Cấu trúc thư mục:
        miotcd/train1/
            car/           <-- tên folder = nhãn type
                img001.jpg
            bus/
                img002.jpg
    """

    print("\n" + "=" * 60)
    print("XỬ LÝ: MIO-TCD  (type = THẬT, color & make = PSEUDO)")
    print("=" * 60)

    # Map tên folder gốc → tên class đã chuẩn hóa (từ prepare_miotcd.py)
    CLASS_MAP = {
    "car":               "car",
    "bus":               "bus",
    "work_van":          "van",
    "pickup_truck":      "pickup",
    "articulated_truck": "truck",
    "single_unit_truck": "truck",
    # Tên đã được processed sẵn trong dataset này
    "pickup":            "pickup",
    "truck":             "truck",
    "van":               "van",
    }

    rows = []  # Sẽ chứa tất cả dòng dữ liệu để ghi ra CSV

    for class_folder in sorted(MIOTCD_PATH.iterdir()):

        if not class_folder.is_dir():
            continue  # Bỏ qua nếu không phải thư mục

        original_class = class_folder.name  # Tên folder gốc, vd: "work_van"

        if original_class not in CLASS_MAP:
            print(f"  Bỏ qua: {original_class} (không phải vehicle)")
            continue

        true_type = CLASS_MAP[original_class]  # Nhãn type thật, vd: "van"

        # Lấy danh sách ảnh trong folder này
        image_files = [
            f for f in class_folder.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"\n  {original_class} → type='{true_type}' | {len(image_files)} ảnh")

        for idx, img_path in enumerate(image_files):

            # In tiến độ mỗi 500 ảnh
            if (idx + 1) % 500 == 0:
                print(f"    ... {idx + 1}/{len(image_files)}")

            # Dự đoán COLOR bằng model color
            pseudo_color, conf_color = predict(model_color, img_path, COLOR_CLASSES)

            # Dự đoán MAKE bằng model make
            pseudo_make, conf_make = predict(model_make, img_path, MAKE_CLASSES)

            rows.append({
                "image_path":  str(img_path),   # Đường dẫn tuyệt đối đến ảnh
                "source":      "miotcd",         # Dataset gốc
                "type":        true_type,        # Nhãn THẬT
                "color":       pseudo_color,     # Nhãn GIẢ từ model
                "make":        pseudo_make,      # Nhãn GIẢ từ model
                "type_conf":   1.0,              # 1.0 vì là nhãn thật
                "color_conf":  round(conf_color, 4),
                "make_conf":   round(conf_make,  4),
            })

    _write_csv(CSV_MIOTCD, rows)
    print(f"\n  Đã lưu {len(rows)} dòng → {CSV_MIOTCD.name}")


def process_color_dataset(model_type: nn.Module, model_make: nn.Module) -> None:
    """
    Xử lý Vehicle Color:
        - COLOR: lấy từ tên folder (nhãn THẬT)
        - TYPE : dùng model_type  dự đoán (PSEUDO)
        - MAKE : dùng model_make  dự đoán (PSEUDO)

    Cấu trúc thư mục:
        vehicle_color/
            train/
                black/       <-- tên folder = nhãn color
                    img.jpg
            val/
            test/
    """

    print("\n" + "=" * 60)
    print("XỬ LÝ: VEHICLE COLOR  (color = THẬT, type & make = PSEUDO)")
    print("=" * 60)

    rows = []

    # Duyệt qua tất cả split (train/val/test)
    for split in ["train", "val", "test"]:

        split_path = COLOR_DATASET_PATH / split

        if not split_path.exists():
            print(f"  Không tìm thấy: {split_path}")
            continue

        for color_folder in sorted(split_path.iterdir()):

            if not color_folder.is_dir():
                continue

            true_color = color_folder.name  # Nhãn màu thật, vd: "black"

            if true_color not in COLOR_CLASSES:
                print(f"  Màu không hợp lệ: '{true_color}' — bỏ qua")
                continue

            image_files = [
                f for f in color_folder.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ]

            print(f"\n  [{split}] {true_color} | {len(image_files)} ảnh")

            for idx, img_path in enumerate(image_files):

                if (idx + 1) % 500 == 0:
                    print(f"    ... {idx + 1}/{len(image_files)}")

                # QUAN TRỌNG: vehicle_color dataset cũng là ảnh chụp gần/đẹp trên mạng
                # Model MIO-TCD sẽ nhận nhầm thành "bus". Ta hardcode "car"
                pseudo_type  = "car"
                conf_type    = 1.0
                
                pseudo_make, conf_make = predict(model_make, img_path, MAKE_CLASSES)

                rows.append({
                    "image_path":  str(img_path),
                    "source":      "vehicle_color",
                    "type":        pseudo_type,   # THẬT (hardcode "car")
                    "color":       true_color,    # THẬT
                    "make":        pseudo_make,   # GIẢ
                    "type_conf":   round(conf_type,  4),
                    "color_conf":  1.0,
                    "make_conf":   round(conf_make,  4),
                })

    _write_csv(CSV_COLOR, rows)
    print(f"\n  Đã lưu {len(rows)} dòng → {CSV_COLOR.name}")


def process_make_dataset(model_type: nn.Module, model_color: nn.Module) -> None:
    """
    Xử lý Vehicle Make (Riotu Cars):
        - MAKE : lấy từ tên folder cấp 1 (nhãn THẬT)
        - TYPE : dùng model_type  dự đoán (PSEUDO)
        - COLOR: dùng model_color dự đoán (PSEUDO)

    Cấu trúc thư mục:
        vehicle_make/
            audi/             <-- tên folder cấp 1 = nhãn make
                A3/           <-- model xe (không quan tâm)
                    img.jpg
    """

    print("\n" + "=" * 60)
    print("XỬ LÝ: VEHICLE MAKE  (make = THẬT, type & color = PSEUDO)")
    print("=" * 60)

    SELECTED_MAKES = [
        "audi", "bmw", "cadillac", "chevrolet", "dodge",
        "ford", "gmc", "honda", "hyundai", "infiniti",
        "jeep", "kia", "landrover", "lexus", "mazda",
        "mercedes", "mitsubishi", "nissan", "porsche", "toyota",
    ]

    rows = []

    for make_name in SELECTED_MAKES:

        make_folder = MAKE_DATASET_PATH / make_name

        if not make_folder.exists():
            print(f"  Không tìm thấy hãng: {make_name}")
            continue

        # rglob("*") tìm đệ quy tất cả file trong tất cả subfolder
        # Cần dùng vì Riotu Cars có cấu trúc: hãng/model/generation/ảnh
        image_files = [
            f for f in make_folder.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"\n  Hãng: {make_name} | {len(image_files)} ảnh")

        for idx, img_path in enumerate(image_files):

            if (idx + 1) % 500 == 0:
                print(f"    ... {idx + 1}/{len(image_files)}")

            # QUAN TRỌNG: vehicle_make dataset CHỈ chứa xe hơi (car)
            # KHÔNG dùng model_type để predict vì model được train trên
            # ảnh camera giám sát (MIO-TCD) → sẽ đoán nhầm ảnh xe đẹp thành "bus"
            # → Hardcode luôn type = "car" để đảm bảo chính xác 100%
            pseudo_type  = "car"
            conf_type    = 1.0

            pseudo_color, conf_color = predict(model_color, img_path, COLOR_CLASSES)

            rows.append({
                "image_path":  str(img_path),
                "source":      "vehicle_make",
                "type":        pseudo_type,    # THẬT (hardcode "car")
                "color":       pseudo_color,   # GIẢ (pseudo-label)
                "make":        make_name,      # THẬT (từ tên folder)
                "type_conf":   round(conf_type,  4),
                "color_conf":  round(conf_color, 4),
                "make_conf":   1.0,
            })

    _write_csv(CSV_MAKE, rows)
    print(f"\n  Đã lưu {len(rows)} dòng → {CSV_MAKE.name}")


# =============================================================================
# PHẦN 7: HÀM TIỆN ÍCH
# =============================================================================

def _write_csv(filepath: Path, rows: list) -> None:
    """Ghi danh sách dict ra file CSV."""

    if not rows:
        print(f"  Không có dữ liệu để ghi vào {filepath.name}")
        return

    # Lấy tên cột từ dict đầu tiên trong danh sách
    fieldnames = list(rows[0].keys())
    # = ["image_path", "source", "type", "color", "make",
    #    "type_conf", "color_conf", "make_conf"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        # DictWriter: ghi CSV từ danh sách dict, tự map key → cột
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()    # Ghi dòng đầu (tên cột)
        writer.writerows(rows)  # Ghi tất cả dòng dữ liệu


def print_statistics(csv_path: Path) -> None:
    """In thống kê của 1 file CSV (tổng dòng, % unknown, phân bố class)."""

    import collections

    print(f"\n  Thống kê: {csv_path.name}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"  Tổng ảnh: {total}")

    # Đếm nhãn "unknown" trong mỗi cột
    for col in ["type", "color", "make"]:
        n_unknown = sum(1 for r in rows if r[col] == "unknown")
        pct = n_unknown / total * 100 if total > 0 else 0
        print(f"  {col:6}: {n_unknown:5} unknown ({pct:.1f}%)")

    # Phân bố TYPE
    type_dist = collections.Counter(r["type"] for r in rows)
    print(f"\n  Phân bố TYPE:")
    for cls, cnt in sorted(type_dist.items()):
        bar = "#" * (cnt // max(1, total // 40))
        print(f"    {cls:10}: {cnt:6}  {bar}")


# =============================================================================
# PHẦN 8: MAIN - CHẠY CHƯƠNG TRÌNH
# =============================================================================

if __name__ == "__main__":
    """
    Chạy bằng lệnh:  python generate_labels.py

    Thứ tự thực hiện:
        1) Load 3 model từ file .pth
        2) Xử lý từng dataset → sinh pseudo-label → lưu CSV
        3) In thống kê kết quả
    """

    print("=" * 60)
    print("GENERATE PSEUDO-LABELS")
    print("=" * 60)

    # Bước 1: Load 3 model
    print("\n[1/4] Đang load các model...")
    model_type  = load_model(MODEL_TYPE_PATH,  num_classes=len(TYPE_CLASSES))   # 5
    model_color = load_model(MODEL_COLOR_PATH, num_classes=len(COLOR_CLASSES))  # 15
    model_make  = load_model(MODEL_MAKE_PATH,  num_classes=len(MAKE_CLASSES))   # 20

    # Bước 2: Xử lý MIO-TCD
    # → MIO-TCD có nhãn TYPE thật, cần sinh pseudo COLOR & MAKE
    print("\n[2/4] Xử lý MIO-TCD...")
    process_miotcd(model_color=model_color, model_make=model_make)

    # Bước 3: Xử lý Vehicle Color
    # → Dataset này có nhãn COLOR thật, cần sinh pseudo TYPE & MAKE
    print("\n[3/4] Xử lý Vehicle Color...")
    process_color_dataset(model_type=model_type, model_make=model_make)

    # Bước 4: Xử lý Vehicle Make
    # → Dataset này có nhãn MAKE thật, cần sinh pseudo TYPE & COLOR
    print("\n[4/4] Xử lý Vehicle Make...")
    process_make_dataset(model_type=model_type, model_color=model_color)

    # In thống kê
    print("\n" + "=" * 60)
    print("THỐNG KÊ KẾT QUẢ")
    print("=" * 60)
    for csv_file in [CSV_MIOTCD, CSV_COLOR, CSV_MAKE]:
        if csv_file.exists():
            print_statistics(csv_file)

    print("\n" + "=" * 60)
    print("HOÀN THÀNH! File CSV đã lưu tại:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)
    print("\nBước tiếp theo:")
    print("  Chạy build_unified_dataset.py để gộp 3 CSV thành 1 bộ train/val/test")
