from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from generate_labels import load_model, predict, COLOR_CLASSES, TYPE_CLASSES, MAKE_CLASSES

# Đường dẫn đến file model color
MODEL_COLOR_PATH = Path(r"D:\TTTN_Co_Phuong\trained_models\best_vehicle_color_resnet50.pth")

# Đường dẫn đến 1 ảnh xe bất kỳ trên máy bạn để test
# (đổi thành ảnh thực tế bạn có)
TEST_IMAGE = Path(r"D:\TTTN_Co_Phuong\audi-a6.jpg")

# ─────────────────────────────────────────────────────────────────

print("=" * 50)
print("KIỂM TRA MODEL COLOR")
print("=" * 50)

# Kiểm tra file tồn tại
if not MODEL_COLOR_PATH.exists():
    print(f"[LỖI] Không tìm thấy model: {MODEL_COLOR_PATH}")
    print("      → Sửa MODEL_COLOR_PATH trong file này cho đúng")
    sys.exit(1)

if not TEST_IMAGE.exists():
    print(f"[LỖI] Không tìm thấy ảnh: {TEST_IMAGE}")
    print("      → Sửa TEST_IMAGE trong file này cho đúng")
    sys.exit(1)

# Load model
print(f"\nLoad model: {MODEL_COLOR_PATH.name}")
model = load_model(MODEL_COLOR_PATH, num_classes=len(COLOR_CLASSES))

# Predict
print(f"Ảnh test:   {TEST_IMAGE.name}")
predicted_class, confidence = predict(model, TEST_IMAGE, COLOR_CLASSES)

print(f"\nKết quả dự đoán:")
print(f"  Màu:        {predicted_class}")
print(f"  Confidence: {confidence:.4f} ({confidence*100:.1f}%)")

print(f"\nDanh sách {len(COLOR_CLASSES)} màu (theo index):")
for i, name in enumerate(COLOR_CLASSES):
    print(f"  [{i:2d}] {name}")

print("\n" + "=" * 50)
print("Nếu kết quả dự đoán hợp lý → class order đúng!")
print("Nếu dự đoán sai màu rõ ràng → class order bị sai, cần kiểm tra lại")
print("=" * 50)
