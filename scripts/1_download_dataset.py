import os
import json
import random
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load env variables
load_dotenv("api.env")

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

DATASET = "abdallahwagih/cars-detection"
DATA_DIR = Path("data/raw")
SAMPLE_DIR = Path("data/samples")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

print("📦 Downloading dataset from Kaggle...")
os.system(f'kaggle datasets download -d {DATASET} -p {DATA_DIR} --unzip')

print("✅ Dataset downloaded successfully.")

# Prepare 10 sample images
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
all_images = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
random.shuffle(all_images)

selected = all_images[:10]
for i, src in enumerate(selected, 1):
    dst = SAMPLE_DIR / f"car_{i:02d}{src.suffix.lower()}"
    shutil.copy2(src, dst)

print(f"✅ Copied {len(selected)} sample images to {SAMPLE_DIR}")
