import os
from pathlib import Path

BASE_DIR = Path(r"D:\TTTN_Co_Phuong\Dataset")

def generate_txt(label_dirs, output_txt, img_replace_func):
    valid_paths = []
    for l_dir in label_dirs:
        if not l_dir.exists():
            continue
        # Quét tất cả file txt trong thư mục (bao gồm cả thư mục con)
        for txt_file in l_dir.rglob("*.txt"):
            if txt_file.name == "classes.txt":
                continue
            
            # Tính toán đường dẫn ảnh gốc tương ứng
            img_path = img_replace_func(txt_file)
            if img_path.exists():
                valid_paths.append(str(img_path) + "\n")
                
    with open(output_txt, "w", encoding="utf-8") as f:
        f.writelines(valid_paths)
    print(f"Đã ghi {len(valid_paths)} đường dẫn vào {output_txt}")

# --- BDD100K ---
def bdd_train_replace(txt_path):
    # D:\...\bdd100k\bdd100k\labels\100k\train\img.txt -> D:\...\bdd100k\bdd100k\images\100k\train\img.jpg
    return Path(str(txt_path).replace("\\labels\\", "\\images\\").replace(".txt", ".jpg"))

# --- UA-DETRAC ---
def detrac_replace(txt_path):
    # D:\...\DETRAC-Labels\MVI_20011\img.txt -> D:\...\DETRAC-Images\DETRAC-Images\MVI_20011\img.jpg
    return Path(str(txt_path).replace("\\DETRAC-Labels\\", "\\DETRAC-Images\\DETRAC-Images\\").replace(".txt", ".jpg"))

if __name__ == "__main__":
    train_txt = BASE_DIR / "train_images.txt"
    val_txt = BASE_DIR / "val_images.txt"
    
    # Tạo list train
    bdd_train_label = BASE_DIR / "bdd100k" / "bdd100k" / "labels" / "100k" / "train"
    detrac_label_dir = BASE_DIR / "DETRAC-Labels"
    generate_txt([bdd_train_label], train_txt, bdd_train_replace)
    
    # Thêm DETRAC vào cuối file train.txt
    detrac_paths = []
    for txt_file in detrac_label_dir.rglob("*.txt"):
        if txt_file.name != "classes.txt":
            img_p = detrac_replace(txt_file)
            if img_p.exists():
                detrac_paths.append(str(img_p) + "\n")
    with open(train_txt, "a", encoding="utf-8") as f:
        f.writelines(detrac_paths)
    print(f"Đã bổ sung {len(detrac_paths)} ảnh UA-DETRAC vào train_images.txt")

    # Tạo list val (chỉ lấy BDD100K val)
    bdd_val_label = BASE_DIR / "bdd100k" / "bdd100k" / "labels" / "100k" / "val"
    generate_txt([bdd_val_label], val_txt, bdd_train_replace)
    
    # Cập nhật dataset.yaml
    yaml_path = BASE_DIR / "dataset.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    with open(yaml_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("train:"):
                f.write(f"train: {train_txt.as_posix()}\n")
            elif line.startswith("val:"):
                f.write(f"val: {val_txt.as_posix()}\n")
            elif not line.strip().startswith("- bdd") and not line.strip().startswith("- DETRAC"):
                f.write(line)
                
    print("Đã cập nhật dataset.yaml để trỏ thẳng vào file .txt, tránh lỗi Background.")
