
from pathlib import Path
import random
import shutil

RAW_DIR = Path("dataset_raw")
OUT_DIR = Path("dataset")

CLASSES = ["baut", "mur"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)

for class_name in CLASSES:
    source = RAW_DIR / class_name

    files = [
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXT
    ]

    random.shuffle(files)
    split_index = int(len(files) * 0.8)

    splits = {
        "train": files[:split_index],
        "val": files[split_index:]
    }

    for split_name, split_files in splits.items():
        target = OUT_DIR / split_name / class_name
        target.mkdir(parents=True, exist_ok=True)

        for old_file in target.iterdir():
            if old_file.is_file():
                old_file.unlink()

        for image_path in split_files:
            shutil.copy2(image_path, target / image_path.name)

        print(f"{split_name}/{class_name}: {len(split_files)} gambar")

print("\nPembagian dataset selesai!")