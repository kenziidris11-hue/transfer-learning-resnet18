
import time
import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Pengaturan
DATA_DIR = Path("dataset")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Perangkat:", DEVICE)

# Persiapan gambar
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train", transform=train_transform
)
val_dataset = datasets.ImageFolder(
    DATA_DIR / "val", transform=val_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

class_names = train_dataset.classes
print("Kelas:", class_names)
print("Jumlah gambar training:", len(train_dataset))
print("Jumlah gambar validation:", len(val_dataset))

# Muat MobileNetV3-Small dengan bobot ImageNet
try:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    print("Bobot pretrained ImageNet berhasil dimuat.")
except Exception as e:
    print("Gagal memuat bobot pretrained:", e)
    print("Periksa koneksi internet, lalu jalankan kembali.")
    raise

# Ganti lapisan output untuk klasifikasi baut dan mur
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(class_names))

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

best_accuracy = 0.0
best_epoch = 0
best_weights = copy.deepcopy(model.state_dict())
history = []
start_time = time.time()

# Training dan validation
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    model.eval()
    val_correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)
    val_loss /= len(val_dataset)
    val_accuracy = val_correct / len(val_dataset)

    scheduler.step()

    history.append({
        "epoch": epoch + 1,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_epoch = epoch + 1
        best_weights = copy.deepcopy(model.state_dict())

training_time = time.time() - start_time
model.load_state_dict(best_weights)

# Simpan model
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "mode": "mobilenet_v3_small"
}, RESULT_DIR / "mobilenet_v3_small.pth")

# Simpan riwayat training
with open(RESULT_DIR / "history_mobilenet.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

# Ukur waktu prediksi per gambar pada CPU
model_cpu = copy.deepcopy(model).to("cpu").eval()
sample = val_dataset[0][0].unsqueeze(0)
repeats = 100

with torch.no_grad():
    for _ in range(10):
        model_cpu(sample)

    start_latency = time.perf_counter()
    for _ in range(repeats):
        model_cpu(sample)
    latency_ms = (time.perf_counter() - start_latency) * 1000 / repeats

# Simpan perbandingan dengan hasil ResNet-18 yang sudah ada
comparison_file = RESULT_DIR / "comparison.csv"

rows = []
if comparison_file.exists():
    with open(comparison_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("mode") != "mobilenet_v3_small"]

rows.append({
    "mode": "mobilenet_v3_small",
    "best_val_accuracy": best_accuracy,
    "best_epoch": best_epoch,
    "training_time_seconds": training_time,
    "latency_ms": latency_ms
})

with open(comparison_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "mode",
            "best_val_accuracy",
            "best_epoch",
            "training_time_seconds",
            "latency_ms"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n=== HASIL MOBILENETV3-SMALL ===")
print(f"Akurasi validation terbaik: {best_accuracy * 100:.2f}%")
print(f"Epoch terbaik: {best_epoch}")
print(f"Waktu training: {training_time / 60:.2f} menit")
print(f"Latency CPU: {latency_ms:.2f} ms/gambar")
print("Model tersimpan: results/mobilenet_v3_small.pth")
print("Perbandingan tersimpan: results/comparison.csv")