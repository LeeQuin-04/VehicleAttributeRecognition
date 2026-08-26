import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from PIL import Image
import io

import google.generativeai as genai

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CAU HINH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_NAME = "gemini-flash-latest"

VALID_TYPES  = {"bus", "car", "pickup", "truck", "van"}
VALID_COLORS = {
    "beige", "black", "blue", "brown", "gold", "green", "grey",
    "orange", "pink", "purple", "red", "silver", "tan", "white", "yellow"
}
VALID_MAKES  = {
    "audi", "bmw", "cadillac", "chevrolet", "dodge", "ford", "gmc",
    "honda", "hyundai", "infiniti", "jeep", "kia", "landrover", "lexus",
    "mazda", "mercedes", "mitsubishi", "nissan", "porsche", "toyota", "unknown"
}

PROMPT = """Look at this vehicle image. Reply ONLY with a JSON object, nothing else:
{"type": "bus|car|pickup|truck|van", "color": "beige|black|blue|brown|gold|green|grey|orange|pink|purple|red|silver|tan|white|yellow", "make": "audi|bmw|cadillac|chevrolet|dodge|ford|gmc|honda|hyundai|infiniti|jeep|kia|landrover|lexus|mazda|mercedes|mitsubishi|nissan|porsche|toyota|unknown"}"""


# =============================================================================
# HAM GOI GEMINI
# =============================================================================

def load_image_pil(image_path: Path) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > 768:
        img.thumbnail((768, 768))
    return img


def call_gemini(model, image_path: Path, retries: int = 3) -> dict | None:
    img = load_image_pil(image_path)
    
    for attempt in range(retries):
        try:
            response = model.generate_content(
                [PROMPT, img],
                generation_config=genai.GenerationConfig(
                    temperature=0.05,
                    max_output_tokens=200,   # Tang len de tranh bi cat ngang
                )
            )
            
            # Kiem tra candidates ton tai
            if not response.candidates:
                print(f"\n  [BLOCKED] Khong co candidates (safety block)")
                return None
            
            candidate = response.candidates[0]
            
            # finish_reason: 1=STOP, 2=MAX_TOKENS, 3=SAFETY
            finish_reason = candidate.finish_reason
            if finish_reason not in (1, 2):  # Bo qua SAFETY va loi khac
                print(f"\n  [SKIP] finish_reason={finish_reason}")
                return None
            
            # Lay text - xu ly an toan
            try:
                text = response.text.strip()
            except ValueError:
                # Response bi cat hoac khong hop le, thu lay truc tiep tu parts
                try:
                    text = candidate.content.parts[0].text.strip()
                except Exception:
                    return None
            
            # Tim JSON trong ket qua tra ve
            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if not json_match:
                # Thu tim JSON khong day du (bi cat), bo qua
                return None
            
            result = json.loads(json_match.group())
            result["type"]  = result.get("type",  "").lower().strip()
            result["color"] = result.get("color", "").lower().strip()
            result["make"]  = result.get("make",  "unknown").lower().strip()
            
            # Validate
            if result["type"]  not in VALID_TYPES:  result["type"]  = "unknown"
            if result["color"] not in VALID_COLORS: result["color"] = "unknown"
            if result["make"]  not in VALID_MAKES:  result["make"]  = "unknown"
            
            return result
            
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"\n  [RATE LIMIT] Doi 60 giay...")
                time.sleep(60)
            elif attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"\n  [LOI] {err_str[:100]}")
                return None
    return None


# =============================================================================
# HAM XU LY CHINH
# =============================================================================

def label_images(api_key: str, image_paths: list, output_csv: Path, rpm: int = 14, resume: bool = True):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Resume: bo qua anh da xu ly
    done_paths = set()
    results = []
    if resume and output_csv.exists():
        df_done = pd.read_csv(output_csv)
        done_paths = set(df_done["image_path"].tolist())
        results = df_done.to_dict("records")
        print(f"[RESUME] Da xu ly {len(done_paths)} anh truoc do.")
    
    todo = [p for p in image_paths if str(p) not in done_paths]
    print(f"Con lai: {len(todo)} anh can xu ly")
    
    delay = 60.0 / rpm
    
    for i, img_path in enumerate(todo):
        print(f"[{i+1}/{len(todo)}] {Path(img_path).name}...", end=" ", flush=True)
        
        label = call_gemini(model, Path(img_path))
        
        if label and label["type"] != "unknown":
            print(f"{label['type']:8} | {label['color']:8} | {label['make']}")
            results.append({
                "image_path": str(img_path),
                "source":     "gemini_labeled",
                "type":       label["type"],
                "color":      label["color"],
                "make":       label["make"],
                "type_conf":  1.0,
                "color_conf": 1.0,
                "make_conf":  1.0,
            })
        else:
            print("SKIP")
        
        # Luu checkpoint moi 50 anh
        if (i + 1) % 50 == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8")
            print(f"  [CHECKPOINT] {len(results)} dong -> {output_csv.name}")
        
        time.sleep(delay)
    
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[HOAN THANH] Luu {len(results)} dong -> {output_csv}")
    return pd.DataFrame(results)


# =============================================================================
# CHAY TU DONG LENH
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--source", default="miotcd_na",
                        choices=["miotcd_na", "color_dataset", "make_dataset"])
    parser.add_argument("--limit",  type=int, default=500)
    parser.add_argument("--rpm",    type=int, default=14)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    out_csv = Path(args.output_csv) if args.output_csv else \
              PROJECT_ROOT / "data" / "pseudo_labeled" / "gemini_labeled.csv"

    if args.source == "miotcd_na":
        df = pd.read_csv(PROJECT_ROOT / "data" / "pseudo_labeled" / "miotcd_labeled.csv")
        df_na = df[df["type"].isin(["bus", "truck", "van"])].head(args.limit)
        image_list = df_na["image_path"].tolist()
        print(f"[miotcd_na] {len(image_list)} anh bus/truck/van")

    elif args.source == "color_dataset":
        p = PROJECT_ROOT / "data" / "raw" / "vehicle_color"
        image_list = [str(f) for f in p.rglob("*.jpg")][:args.limit]
        print(f"[color_dataset] {len(image_list)} anh")

    elif args.source == "make_dataset":
        p = PROJECT_ROOT / "data" / "raw" / "vehicle_make"
        image_list = [str(f) for f in p.rglob("*.jpg")][:args.limit]
        print(f"[make_dataset] {len(image_list)} anh")

    print(f"Toc do: {args.rpm} req/phut | Output: {out_csv}\n")
    label_images(args.api_key, image_list, out_csv, rpm=args.rpm)
