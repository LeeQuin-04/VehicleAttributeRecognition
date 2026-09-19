import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = Path(r"D:\TTTN_Co_Phuong\Dataset")
BDD_IMG_DIR = BASE_DIR / "bdd100k" / "bdd100k" / "images" / "100k"
BDD_LABEL_JSON_TRAIN = BASE_DIR / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"
BDD_LABEL_JSON_VAL = BASE_DIR / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

DETRAC_IMG_DIR = BASE_DIR / "DETRAC-Images" / "DETRAC-Images"
DETRAC_XML_TRAIN_DIR = BASE_DIR / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML"
DETRAC_XML_TEST_DIR = BASE_DIR / "DETRAC-Test-Annotations-XML" / "DETRAC-Test-Annotations-XML"

# --- TARGET CLASSES ---
# 0: car, 1: bus, 2: truck, 3: motorbike, 4: bicycle
CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "motorcycle": 3,
    "bicycle": 4,
    "van": 0 # Gộp van của UA-DETRAC vào car
}

def convert_box_to_yolo(box_width, box_height, xmin, ymin, img_width, img_height):
    xmax = xmin + box_width
    ymax = ymin + box_height
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    w = box_width / img_width
    h = box_height / img_height
    return x_center, y_center, w, h

def convert_box_to_yolo_bdd(x1, y1, x2, y2, img_width, img_height):
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return x_center, y_center, w, h

def process_bdd(json_path, split):
    print(f"\nĐang xử lý BDD100K ({split})...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Tạo thư mục labels song song với images
    # images: bdd100k/bdd100k/images/100k/train
    # labels: bdd100k/bdd100k/labels/100k/train
    label_dir = BASE_DIR / "bdd100k" / "bdd100k" / "labels" / "100k" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    
    img_width, img_height = 1280, 720 # Default BDD100K resolution
    
    count = 0
    for item in tqdm(data):
        img_name = item["name"]
        labels = item.get("labels", [])
        
        yolo_lines = []
        for lbl in labels:
            if "box2d" not in lbl:
                continue
            cat = lbl["category"]
            if cat in CLASS_MAP:
                cls_id = CLASS_MAP[cat]
                box = lbl["box2d"]
                x_c, y_c, w, h = convert_box_to_yolo_bdd(box["x1"], box["y1"], box["x2"], box["y2"], img_width, img_height)
                yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        
        if yolo_lines:
            txt_path = label_dir / img_name.replace(".jpg", ".txt")
            with open(txt_path, "w") as f:
                f.writelines(yolo_lines)
            count += 1
            
    print(f"Hoàn thành BDD100K {split}: {count} ảnh hợp lệ.")
    return label_dir

def process_detrac(xml_dir, img_base_dir, is_train=True):
    print(f"\nĐang xử lý UA-DETRAC ({'Train' if is_train else 'Test'})...")
    # Tương tự, YOLO sẽ tìm labels bằng cách thay images -> labels
    label_base_dir = BASE_DIR / "DETRAC-Labels"
    
    if not xml_dir.exists():
        print(f"Không tìm thấy {xml_dir}")
        return None
        
    count = 0
    for xml_file in tqdm(list(xml_dir.glob("*.xml"))):
        seq_name = xml_file.stem
        img_seq_dir = img_base_dir / seq_name
        label_seq_dir = label_base_dir / seq_name
        
        if not img_seq_dir.exists():
            continue
            
        label_seq_dir.mkdir(parents=True, exist_ok=True)
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # UA-DETRAC độ phân giải 960x540
        img_width, img_height = 960, 540
        
        for frame in root.findall(".//frame"):
            frame_idx = int(frame.get("num"))
            
            # CHỈ LẤY MẪU 1/10 FRAME (để tránh lặp dữ liệu quá nhiều)
            if frame_idx % 10 != 0:
                continue
                
            yolo_lines = []
            target_list = frame.find("target_list")
            if target_list is not None:
                for target in target_list.findall("target"):
                    obj_type = target.find("attribute").get("vehicle_type")
                    if obj_type in CLASS_MAP:
                        cls_id = CLASS_MAP[obj_type]
                        box = target.find("box")
                        w = float(box.get("width"))
                        h = float(box.get("height"))
                        x = float(box.get("left"))
                        y = float(box.get("top"))
                        x_c, y_c, w_n, h_n = convert_box_to_yolo(w, h, x, y, img_width, img_height)
                        yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                        
            if yolo_lines:
                img_name = f"img{frame_idx:05d}.jpg"
                txt_path = label_seq_dir / img_name.replace(".jpg", ".txt")
                with open(txt_path, "w") as f:
                    f.writelines(yolo_lines)
                count += 1
                
    print(f"Hoàn thành UA-DETRAC: {count} frames đã lấy mẫu.")
    return label_base_dir

if __name__ == "__main__":
    # 1. BDD100K
    if BDD_LABEL_JSON_TRAIN.exists():
        process_bdd(BDD_LABEL_JSON_TRAIN, "train")
    if BDD_LABEL_JSON_VAL.exists():
        process_bdd(BDD_LABEL_JSON_VAL, "val")
        
    # 2. UA-DETRAC
    process_detrac(DETRAC_XML_TRAIN_DIR, DETRAC_IMG_DIR, is_train=True)
    process_detrac(DETRAC_XML_TEST_DIR, DETRAC_IMG_DIR, is_train=False)
    
    # 3. Tạo file dataset.yaml
    yaml_content = f"""path: {BASE_DIR.as_posix()}
train:
  - bdd100k/bdd100k/images/100k/train
  - DETRAC-Images # YOLO sẽ đệ quy quét ảnh và tự map sang DETRAC-Labels
val:
  - bdd100k/bdd100k/images/100k/val

names:
  0: car
  1: bus
  2: truck
  3: motorbike
  4: bicycle
"""
    yaml_path = BASE_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"\n[XONG] File cấu hình YOLO được lưu tại: {yaml_path}")
    print("Bạn có thể bắt đầu train YOLO bằng lệnh:")
    print(f"yolo task=detect mode=train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640")
