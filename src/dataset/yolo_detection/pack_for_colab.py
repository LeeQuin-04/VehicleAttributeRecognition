import os
import zipfile
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(r"D:\TTTN_Co_Phuong\Dataset")
TRAIN_TXT = BASE_DIR / "train_images.txt"
VAL_TXT = BASE_DIR / "val_images.txt"
OUTPUT_ZIP = BASE_DIR / "YOLO_Dataset_Colab.zip"

def get_label_path_bdd(img_path_str):
    # D:\...\bdd100k\bdd100k\images\100k\train\img.jpg -> labels\100k\train\img.txt
    return img_path_str.replace("\\images\\", "\\labels\\").replace(".jpg", ".txt")

def get_label_path_detrac(img_path_str):
    # D:\...\DETRAC-Images\DETRAC-Images\MVI_20011\img.jpg -> DETRAC-Labels\MVI_20011\img.txt
    return img_path_str.replace("\\DETRAC-Images\\DETRAC-Images\\", "\\DETRAC-Labels\\").replace(".jpg", ".txt")

def add_files_to_zip(txt_file, split_name, zipf):
    with open(txt_file, "r", encoding="utf-8") as f:
        img_paths = [line.strip() for line in f.readlines()]
        
    print(f"\nĐang nén {len(img_paths)} ảnh cho tập {split_name}...")
    
    for img_path_str in tqdm(img_paths):
        img_path = Path(img_path_str)
        if not img_path.exists():
            continue
            
        # Xác định đường dẫn file label
        if "bdd100k" in img_path_str:
            lbl_path = Path(get_label_path_bdd(img_path_str))
        else:
            lbl_path = Path(get_label_path_detrac(img_path_str))
            
        # Tên file bên trong ZIP:
        # Cấu trúc: 
        # YOLO_Dataset/images/train/img_name.jpg
        # YOLO_Dataset/labels/train/img_name.txt
        # Để tránh trùng tên giữa BDD và DETRAC, ta giữ lại tên thư mục cha
        parent_name = img_path.parent.name
        arc_img = f"YOLO_Dataset/images/{split_name}/{parent_name}_{img_path.name}"
        arc_lbl = f"YOLO_Dataset/labels/{split_name}/{parent_name}_{lbl_path.name}"
        
        # Thêm vào zip
        zipf.write(img_path, arc_img)
        if lbl_path.exists():
            zipf.write(lbl_path, arc_lbl)

if __name__ == "__main__":
    print(f"Bắt đầu đóng gói dữ liệu tại: {OUTPUT_ZIP}")
    
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_STORED) as zipf:
        if TRAIN_TXT.exists():
            add_files_to_zip(TRAIN_TXT, "train", zipf)
        if VAL_TXT.exists():
            add_files_to_zip(VAL_TXT, "val", zipf)
            
        # Thêm file dataset.yaml dành riêng cho Colab
        yaml_colab = """path: /content/YOLO_Dataset
train: images/train
val: images/val

names:
  0: car
  1: bus
  2: truck
  3: motorbike
  4: bicycle
"""
        zipf.writestr("YOLO_Dataset/dataset.yaml", yaml_colab)
        
    print(f"\n[HOÀN THÀNH] Đã nén xong toàn bộ dataset. Dung lượng ZIP: {os.path.getsize(OUTPUT_ZIP) / (1024*1024*1024):.2f} GB")
