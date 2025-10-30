import os
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split



# --- Paths (✅ adjust as needed) ---
EXPORT_FILE = Path(r"C:\Desktop\vision_agent\exports\car_dataset_export_7eemi9UeFzb4aoHe7T1n.json")
IMAGE_SOURCE_DIR = Path(r"C:\Desktop\vision_agent\data\samples")
YOLO_DATA_DIR = Path(r"C:\Desktop\vision_agent\data\processed\yolo_car_dataset")

# --- Single Class ---
CLASS_NAMES = ["Car"]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# --- Reset dataset folder ---
if YOLO_DATA_DIR.exists():
    shutil.rmtree(YOLO_DATA_DIR)

for split in ["train", "val"]:
    (YOLO_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# --- Load JSON ---
if not EXPORT_FILE.exists():
    raise FileNotFoundError(f"❌ Export file not found: {EXPORT_FILE}")

with open(EXPORT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} items from {EXPORT_FILE}")

# --- Split Train/Val ---
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# ===============================
# 🔄 Conversion Function
# ===============================
def convert_and_save(items, split):
    print(f"\n📂 Processing {split} split ({len(items)} images)...")

    for item in items:
        file_name = item["file_name"]
        width = item["file_metadata"]["image_width"]
        height = item["file_metadata"]["image_height"]

        # --- Copy image ---
        src = IMAGE_SOURCE_DIR / file_name
        dst = YOLO_DATA_DIR / "images" / split / file_name
        if not src.exists():
            print(f"⚠️ Missing image: {src}")
            continue
        shutil.copy(src, dst)

        # --- Prepare label file ---
        label_path = YOLO_DATA_DIR / "labels" / split / f"{Path(file_name).stem}.txt"
        lines = []

        for ans_group in item.get("latest_answer", []):
            for ann in ans_group.get("answer", []):
                label = ann.get("label")
                if label != "Car":  # ✅ Only keep Car detections
                    continue

                cls_id = CLASS_MAP["Car"]
                bbox = ann["answer"]

                # --- YOLO normalized format ---
                x_center = ((bbox["xmin"] + bbox["xmax"]) / 2) / width
                y_center = ((bbox["ymin"] + bbox["ymax"]) / 2) / height
                w = (bbox["xmax"] - bbox["xmin"]) / width
                h = (bbox["ymax"] - bbox["ymin"]) / height

                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Save labels
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"✅ Done creating {split} data with labels and images.")

# --- Run Conversion ---
convert_and_save(train_data, "train")
convert_and_save(val_data, "val")

print("\n🎯 YOLO dataset successfully created at:", YOLO_DATA_DIR)
print("📁 Classes:", CLASS_MAP)
print("🚀 Ready to train your YOLOv8 model!")
