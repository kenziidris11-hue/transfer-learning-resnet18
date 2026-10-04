
import csv
from pathlib import Path
from PIL import Image

# Folder dataset gambar asli
DATASET_DIR = Path("dataset_raw")
OUTPUT_FILE = DATASET_DIR / "metadata.csv"

# Format gambar yang didukung
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

rows = []

# Membaca setiap kelas, misalnya baut dan mur
for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    label = class_dir.name

    for image_path in sorted(class_dir.rglob("*")):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format

            rows.append({
                "filename": image_path.name,
                "filepath": image_path.as_posix(),
                "label": label,
                "width": width,
                "height": height,
                "format": image_format
            })

        except Exception as error:
            print(f"Gambar dilewati: {image_path} - {error}")

# Menyimpan metadata ke CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
    columns = [
        "filename",
        "filepath",
        "label",
        "width",
        "height",
        "format"
    ]

    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("\n=== METADATA DATASET SELESAI ===")
print(f"Jumlah gambar tercatat: {len(rows)}")
print(f"File tersimpan: {OUTPUT_FILE}")
print("\nJumlah gambar per kelas:")

for label in sorted(set(row["label"] for row in rows)):
    total = sum(1 for row in rows if row["label"] == label)
    print(f"{label}: {total} gambar")