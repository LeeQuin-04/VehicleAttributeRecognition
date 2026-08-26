from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from generate_labels import (
    load_model, predict,
    MODEL_TYPE_PATH, MODEL_COLOR_PATH, MODEL_MAKE_PATH,
    TYPE_CLASSES, COLOR_CLASSES, MAKE_CLASSES,
)

# ─── SỬA ĐƯỜNG DẪN ẢNH TẠI ĐÂY ───────────────────────────
TEST_IMAGE_PATH = Path(r"D:\TTTN_Co_Phuong\bus.jpg")
# ──────────────────────────────────────────────────────────

print("=" * 50)
print("LOADING MODELS...")
print("=" * 50)

model_type  = load_model(MODEL_TYPE_PATH,  num_classes=len(TYPE_CLASSES))
model_color = load_model(MODEL_COLOR_PATH, num_classes=len(COLOR_CLASSES))
model_make  = load_model(MODEL_MAKE_PATH,  num_classes=len(MAKE_CLASSES))

print("\n" + "=" * 50)
print(f"ẢNH: {TEST_IMAGE_PATH.name}")
print("=" * 50)

pred_type,  conf_type  = predict(model_type,  TEST_IMAGE_PATH, TYPE_CLASSES)
pred_color, conf_color = predict(model_color, TEST_IMAGE_PATH, COLOR_CLASSES)
# pred_make,  conf_make  = predict(model_make,  TEST_IMAGE_PATH, MAKE_CLASSES)

# Chỉ chạy make model nếu là car hoặc pickup
TYPES_WITH_MAKE = {"car", "pickup"}
if pred_type in TYPES_WITH_MAKE:
    pred_make, conf_make = predict(model_make, TEST_IMAGE_PATH, MAKE_CLASSES)
else:
    pred_make, conf_make = "N/A", 1.0   # bus/truck/van không có hãng

print(f"\n  TYPE  : {pred_type:10}  ({conf_type*100:.1f}%)")
print(f"  COLOR : {pred_color:10}  ({conf_color*100:.1f}%)")
print(f"  MAKE  : {pred_make:10}  ({conf_make*100:.1f}%)")
print()