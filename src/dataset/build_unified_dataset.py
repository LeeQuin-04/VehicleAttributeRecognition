import json
import random
from pathlib import Path

import pandas as pd 


# =============================================================================
# CẤU HÌNH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# __file__ = src/dataset/build_unified_dataset.py
# .parent.parent.parent = VehicleAttributeRecognition/ (thư mục project)

# -- Đường dẫn input (3 CSV từ generate_labels.py) ----------------------------
PSEUDO_DIR = PROJECT_ROOT / "data" / "pseudo_labeled"
CSV_MIOTCD = PSEUDO_DIR / "miotcd_labeled.csv"
CSV_COLOR  = PSEUDO_DIR / "color_labeled.csv"
CSV_MAKE   = PSEUDO_DIR / "make_labeled.csv"
CSV_GEMINI = PSEUDO_DIR / "gemini_labeled.csv"  # <-- Thêm file Gemini

# -- Đường dẫn output ---------------------------------------------------------
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"
UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TRAIN = UNIFIED_DIR / "train.csv"
OUTPUT_VAL   = UNIFIED_DIR / "val.csv"
OUTPUT_TEST  = UNIFIED_DIR / "test.csv"
OUTPUT_INFO  = UNIFIED_DIR / "class_info.json"  # Lưu mapping class → index

# -- Tỷ lệ chia ---------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# -- Seed để kết quả chia luôn giống nhau mỗi lần chạy -----------------------
RANDOM_SEED = 42

# -- Danh sách class hợp lệ (khớp với generate_labels.py) --------------------
VALID_TYPES  = {"bus", "car", "pickup", "truck", "van"}
VALID_COLORS = {
    "beige", "black", "blue", "brown", "gold",
    "green", "grey", "orange", "pink", "purple",
    "red", "silver", "tan", "white", "yellow",
}
VALID_MAKES = {
    "audi", "bmw", "cadillac", "chevrolet", "dodge",
    "ford", "gmc", "honda", "hyundai", "infiniti",
    "jeep", "kia", "landrover", "lexus", "mazda",
    "mercedes", "mitsubishi", "nissan", "porsche", "toyota",
}

TYPES_WITH_MAKE    = {"car", "pickup"}
TYPES_WITHOUT_MAKE = {"bus", "truck", "van"}


# =============================================================================
# HÀM ĐỌC VÀ LỌC 1 CSV
# =============================================================================

def load_and_filter(csv_path: Path, source_name: str) -> pd.DataFrame:
    """
    Đọc 1 file CSV pseudo-label và lọc theo FILTER_STRATEGY.

    Tham số:
        csv_path    (Path): đường dẫn đến file CSV
        source_name (str) : tên dataset ("miotcd"/"vehicle_color"/"vehicle_make")

    Trả về:
        df (DataFrame): bảng dữ liệu đã lọc, chỉ chứa cột cần thiết
    """

    print(f"\n  Đọc: {csv_path.name}")

    # Đọc CSV vào DataFrame
    # DataFrame = bảng dữ liệu 2 chiều, giống Excel
    # Mỗi cột = 1 Series, truy cập bằng df["tên_cột"]
    df = pd.read_csv(csv_path)
    print(f"  Tổng dòng ban đầu: {len(df)}")

    # -- Kiểm tra cột tồn tại -------------------------------------------------
    required_cols = ["image_path", "source", "type", "color", "make"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV thiếu cột: '{col}' trong {csv_path.name}")

    # Nhóm 1: car/pickup → cần đủ cả 3 nhãn hợp lệ (type + color + make)
    mask_with_make = (
        df["type"].isin(TYPES_WITH_MAKE) &
        df["color"].isin(VALID_COLORS) &
        df["make"].isin(VALID_MAKES)
    )

    # Nhóm 2: bus/truck/van → chỉ cần type + color hợp lệ, make không quan tâm
    mask_without_make = (
        df["type"].isin(TYPES_WITHOUT_MAKE) &
        df["color"].isin(VALID_COLORS)
    )

    # Gộp 2 nhóm: giữ tất cả dòng thỏa 1 trong 2 điều kiện
    df_filtered = df[mask_with_make | mask_without_make].copy()
    # .copy() để tránh SettingWithCopyWarning khi sửa DataFrame sau này

    # Gán "N/A" cho cột make của bus/truck/van
    # Thay thế toàn bộ pseudo-label make (thường sai) bằng "N/A"
    # "N/A" → index = -1 → training script sẽ bỏ qua loss make cho những ảnh này
    df_filtered.loc[
        df_filtered["type"].isin(TYPES_WITHOUT_MAKE), "make"
    ] = "N/A"

    n_removed = len(df) - len(df_filtered)
    print(f"  Đã lọc: {n_removed} dòng bị loại | Còn lại: {len(df_filtered)} dòng")

    # -- Chỉ giữ các cột cần thiết --------------------------------------------
    # Bỏ các cột confidence (type_conf, color_conf, make_conf)
    # vì khi train model mới, mình không dùng đến chúng nữa
    df_filtered = df_filtered[required_cols].reset_index(drop=True)
    # reset_index(drop=True) → đánh lại chỉ số hàng từ 0

    return df_filtered


# =============================================================================
# HÀM GỘP 3 DATAFRAME
# =============================================================================

def merge_datasets(df_miotcd: pd.DataFrame,
                   df_color:  pd.DataFrame,
                   df_make:   pd.DataFrame) -> pd.DataFrame:
    """
    Gộp 3 DataFrame thành 1, in thống kê tổng quan.

    Trả về:
        df_merged (DataFrame): bảng gộp, đã xáo trộn ngẫu nhiên
    """

    print("\n" + "=" * 60)
    print("BƯỚC 2: GỘP 3 DATASET")
    print("=" * 60)

    # pd.concat() nối các DataFrame theo chiều dọc (axis=0 = thêm hàng)
    # ignore_index=True → đánh lại chỉ số từ 0 sau khi gộp
    df_merged = pd.concat(
        [df_miotcd, df_color, df_make],
        axis=0,
        ignore_index=True,
    )

    print(f"\n  Tổng ảnh sau khi gộp: {len(df_merged)}")
    print(f"    - MIO-TCD:      {len(df_miotcd)}")
    print(f"    - Vehicle Color:{len(df_color)}")
    print(f"    - Vehicle Make: {len(df_make)}")

    # In phân bố từng task
    print(f"\n  Phân bố TYPE ({df_merged['type'].nunique()} classes):")
    # value_counts() đếm số lần xuất hiện của mỗi giá trị, sắp xếp giảm dần
    for cls, cnt in df_merged["type"].value_counts().items():
        print(f"    {cls:12}: {cnt:6}")

    print(f"\n  Phân bố COLOR ({df_merged['color'].nunique()} classes):")
    for cls, cnt in df_merged["color"].value_counts().items():
        print(f"    {cls:12}: {cnt:6}")

    print(f"\n  Phân bố MAKE ({df_merged['make'].nunique()} classes):")
    for cls, cnt in df_merged["make"].value_counts().items():
        print(f"    {cls:12}: {cnt:6}")

    # Giới hạn số lượng ảnh mỗi class để tránh mất cân bằng (Bias)
    MAX_SAMPLES_PER_TYPE = 2500

    def cap_by_class(df, task, max_samples):
        if max_samples is None:
            return df
        
        # Tạo danh sách df con cho mỗi group sau đó dùng concat để nối lại
        frames = []
        for name, group in df.groupby(task):
            frames.append(group.sample(min(len(group), max_samples), random_state=RANDOM_SEED))
        return pd.concat(frames).reset_index(drop=True)

    # Cắt gọn task 'type' (giảm bus/truck xuống để ngang với car)
    print(f"\n  Cắt gọn (Cap) dữ liệu TYPE tối đa {MAX_SAMPLES_PER_TYPE} ảnh/class...")
    df_merged = cap_by_class(df_merged, "type", MAX_SAMPLES_PER_TYPE)

    print(f"  Tổng ảnh sau khi Cân bằng: {len(df_merged)}")

    # Xáo trộn ngẫu nhiên toàn bộ dataset
    # Lý do: khi chia train/val/test theo tỷ lệ, cần đảm bảo ngẫu nhiên
    df_merged = df_merged.sample(
        frac=1,             # frac=1 = lấy 100% dòng (tức là xáo trộn toàn bộ)
        random_state=RANDOM_SEED,  # Seed cố định → kết quả giống nhau mỗi lần chạy
    ).reset_index(drop=True)

    return df_merged


# =============================================================================
# HÀM CHIA TRAIN / VAL / TEST
# =============================================================================

def split_dataset(df: pd.DataFrame) -> tuple:
    """
    Chia DataFrame thành 3 phần: train, val, test.

    Chiến lược chia: Stratified by SOURCE
    → Đảm bảo mỗi split (train/val/test) có tỷ lệ ảnh từ mỗi dataset tương đương.
    → Tránh trường hợp val/test chỉ toàn ảnh từ 1 dataset.

    Ví dụ nếu tổng có 10.000 ảnh (50% miotcd, 30% color, 20% make):
        train (70%): 3500 miotcd + 2100 color + 1400 make
        val   (15%):  750 miotcd +  450 color +  300 make
        test  (15%):  750 miotcd +  450 color +  300 make

    Trả về:
        (df_train, df_val, df_test): 3 DataFrame
    """

    print("\n" + "=" * 60)
    print("BƯỚC 3: CHIA TRAIN / VAL / TEST")
    print("=" * 60)

    train_parts = []  # Sẽ chứa phần train của mỗi source
    val_parts   = []  # Sẽ chứa phần val   của mỗi source
    test_parts  = []  # Sẽ chứa phần test  của mỗi source

    # df.groupby("source") → nhóm các dòng có cùng giá trị "source"
    # Kết quả: iterator trả về (tên_source, sub_dataframe)
    for source_name, group in df.groupby("source"):

        n = len(group)  # Tổng số ảnh của source này

        # Tính điểm cắt
        train_end = int(n * TRAIN_RATIO)
        val_end   = train_end + int(n * VAL_RATIO)
        # Phần còn lại tự động là test

        # Cắt theo index
        # Lưu ý: group đã được xáo trộn từ bước merge_datasets
        # nên cắt thẳng là ổn
        train_part = group.iloc[:train_end]         # iloc = lấy theo vị trí số
        val_part   = group.iloc[train_end:val_end]
        test_part  = group.iloc[val_end:]

        print(f"\n  {source_name}:")
        print(f"    Tổng: {n} | Train: {len(train_part)} | Val: {len(val_part)} | Test: {len(test_part)}")

        train_parts.append(train_part)
        val_parts.append(val_part)
        test_parts.append(test_part)

    # Gộp lại và xáo trộn thêm 1 lần nữa (tránh các ảnh cùng source nằm liền nhau)
    df_train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df_val   = pd.concat(val_parts,   ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df_test  = pd.concat(test_parts,  ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n  Kết quả cuối:")
    print(f"    Train: {len(df_train)} ảnh ({len(df_train)/len(df)*100:.1f}%)")
    print(f"    Val:   {len(df_val)}   ảnh ({len(df_val)/len(df)*100:.1f}%)")
    print(f"    Test:  {len(df_test)}  ảnh ({len(df_test)/len(df)*100:.1f}%)")

    return df_train, df_val, df_test


# =============================================================================
# HÀM TẠO MAPPING CLASS → INDEX
# =============================================================================

def save_class_info(df_train: pd.DataFrame) -> dict:
    """
    Tạo và lưu file class_info.json — mapping từ tên class → số index.

    Ví dụ:
        {
            "type":  {"bus": 0, "car": 1, "pickup": 2, "truck": 3, "van": 4},
            "color": {"beige": 0, "black": 1, ...},
            "make":  {"audi": 0, "bmw": 1, ...}
        }

    File này cần thiết khi:
        - Xây dựng PyTorch Dataset (chuyển tên class → số để tính loss)
        - Inference (chuyển số dự đoán → tên class để hiển thị)
    """

    print("\n" + "=" * 60)
    print("BƯỚC 4: TẠO MAPPING CLASS → INDEX")
    print("=" * 60)

    class_info = {}

    # Dùng df_train để lấy danh sách class (tránh class lạ xuất hiện trong val/test)
    for task in ["type", "color", "make"]:

        # Lấy tất cả giá trị unique, loại bỏ "unknown" và "N/A", sort alphabet
        # "N/A" sẽ được xử lý riêng bên dưới (không phải 1 class thật)
        classes = sorted([
            c for c in df_train[task].unique()
            if c not in ("unknown", "N/A")
        ])
        # sorted() sắp xếp alphabet → index nhất quán mỗi lần chạy

        # Tạo dict: tên class → số index
        # enumerate(classes) trả về (0, "bus"), (1, "car"), ...
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # Thêm "N/A" với index = -1 (chỉ có ý nghĩa với task "make")
        # -1 là sentinel: training script nhìn thấy -1 → bỏ qua loss make
        if task == "make":
            class_to_idx["N/A"] = -1

        class_info[task] = class_to_idx

        n_na = (df_train[task] == "N/A").sum()
        if task == "make":
            print(f"\n  {task.upper()} ({len(classes)} classes thật + {n_na} ảnh N/A):")
        else:
            print(f"\n  {task.upper()} ({len(classes)} classes):")
        print(f"    {class_to_idx}")

    # Lưu ra file JSON để dùng lại sau
    with open(OUTPUT_INFO, "w", encoding="utf-8") as f:
        json.dump(class_info, f, indent=2, ensure_ascii=False)
    # indent=2 → format đẹp, mỗi cấp thụt 2 dấu cách
    # ensure_ascii=False → giữ nguyên ký tự unicode (nếu có)

    print(f"\n  Đã lưu class_info.json → {OUTPUT_INFO}")
    return class_info


# =============================================================================
# IN THỐNG KÊ CUỐI
# =============================================================================

def print_final_stats(df_train, df_val, df_test):
    """In bảng thống kê phân bố class trong từng split."""

    print("\n" + "=" * 60)
    print("THỐNG KÊ PHÂN BỐ CLASS")
    print("=" * 60)

    for task in ["type", "color", "make"]:
        print(f"\n  {task.upper()}:")
        print(f"  {'CLASS':15} {'TRAIN':>8} {'VAL':>8} {'TEST':>8} {'TOTAL':>8}")
        print(f"  {'-'*47}")

        # Lấy tất cả class có trong task này
        all_classes = sorted(set(
            list(df_train[task].unique()) +
            list(df_val[task].unique()) +
            list(df_test[task].unique())
        ) - {"unknown"})

        for cls in all_classes:
            # Đếm số dòng có giá trị = cls trong từng split
            n_train = (df_train[task] == cls).sum()
            n_val   = (df_val[task]   == cls).sum()
            n_test  = (df_test[task]  == cls).sum()
            n_total = n_train + n_val + n_test
            print(f"  {cls:15} {n_train:>8} {n_val:>8} {n_test:>8} {n_total:>8}")


# =============================================================================
# CHẠY CHƯƠNG TRÌNH CHÍNH
# =============================================================================

if __name__ == "__main__":
    """
    Chạy bằng lệnh:
        python build_unified_dataset.py
    """

    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("BUILD UNIFIED DATASET")
    print("=" * 60)

    # -- Bước 1: Đọc và lọc 3 CSV --------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 1: ĐỌC VÀ LỌC DỮ LIỆU")
    print("=" * 60)

    # Kiểm tra 3 file CSV tồn tại
    for path in [CSV_MIOTCD, CSV_COLOR, CSV_MAKE]:
        if not path.exists():
            print(f"\n[LỖI] Không tìm thấy: {path}")
            print("      → Chạy generate_labels.py trước!")
            raise SystemExit(1)

    df_miotcd = load_and_filter(CSV_MIOTCD, "miotcd")
    df_color  = load_and_filter(CSV_COLOR,  "vehicle_color")
    df_make   = load_and_filter(CSV_MAKE,   "vehicle_make")

    # -- Bước 2: Gộp 3 DataFrame ----------------------------------------------
    df_all = merge_datasets(df_miotcd, df_color, df_make)

    # -- Bước 3: Chia train/val/test -------------------------------------------
    df_train, df_val, df_test = split_dataset(df_all)

    # -- Bước 4: Lưu mapping class → index ------------------------------------
    class_info = save_class_info(df_train)

    # -- Bước 5: Lưu 3 file CSV -----------------------------------------------
    df_train.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8")
    df_val.to_csv(OUTPUT_VAL,     index=False, encoding="utf-8")
    df_test.to_csv(OUTPUT_TEST,   index=False, encoding="utf-8")
    # index=False → không lưu cột index (số thứ tự) vào CSV

    print(f"\n  Đã lưu:")
    print(f"    {OUTPUT_TRAIN.name}: {len(df_train)} dòng")
    print(f"    {OUTPUT_VAL.name}:   {len(df_val)} dòng")
    print(f"    {OUTPUT_TEST.name}:  {len(df_test)} dòng")

    # -- Bước 6: In thống kê ---------------------------------------------------
    print_final_stats(df_train, df_val, df_test)

    print("\n" + "=" * 60)
    print("HOÀN THÀNH! Dataset đã lưu tại:")
    print(f"  {UNIFIED_DIR}")
    print("=" * 60)
    print("\nBước tiếp theo:")
    print("  Viết vehicle_dataset.py (PyTorch Dataset class)")
    print("  để đọc train.csv / val.csv / test.csv khi train model")
