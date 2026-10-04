
import copy
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# =========================
# PENGATURAN
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
RESULT_DIR = BASE_DIR / "results"

EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("Perangkat:", DEVICE)
print("Membaca dataset dari:", DATA_DIR)


# =========================
# DATASET DAN AUGMENTASI
# =========================
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
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train", transform=train_transform
)
val_dataset = datasets.ImageFolder(
    DATA_DIR / "val", transform=val_transform
)

if train_dataset.classes != val_dataset.classes:
    raise ValueError("Kelas train dan val tidak sama.")

class_names = train_dataset.classes
num_classes = len(class_names)

if num_classes < 2:
    raise ValueError("Dataset harus memiliki minimal dua kelas.")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS
)

print("Kelas:", class_names)
print("Jumlah gambar train:", len(train_dataset))
print("Jumlah gambar val:", len(val_dataset))


# =========================
# MEMBUAT MODEL
# =========================
def create_model(mode):
    if mode == "scratch":
        model = models.resnet18(weights=None)
    else:
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if mode == "feature":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif mode == "partial":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif mode == "scratch":
        for parameter in model.parameters():
            parameter.requires_grad = True

    return model.to(DEVICE)


# =========================
# MELATIH DAN MENGUJI
# =========================
def run_training(mode):
    print(f"\n{'=' * 45}")
    print("Memulai mode:", mode)

    model = create_model(mode)
    criterion = nn.CrossEntropyLoss()

    if mode == "feature":
        learning_rate = 1e-3
    elif mode == "partial":
        learning_rate = 1e-4
    else:
        learning_rate = 1e-3

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_accuracy = 0.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    history = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Fase training
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_total = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        # Fase validasi
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_total = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        train_loss = train_loss_total / train_total
        val_loss = val_loss_total / val_total

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

        elapsed_epoch = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train: {train_accuracy * 100:.2f}% | "
            f"Val: {val_accuracy * 100:.2f}% | "
            f"Waktu: {elapsed_epoch:.1f} detik"
        )

    training_time = time.time() - start_time

    model.load_state_dict(best_weights)
    model.eval()

    model_path = RESULT_DIR / f"resnet18_{mode}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "mode": mode,
    }, model_path)

    # Mengukur waktu prediksi satu gambar
    sample, _ = val_dataset[0]
    sample = sample.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for _ in range(3):
            model(sample)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        latency_start = time.perf_counter()
        for _ in range(20):
            model(sample)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        latency_ms = (
            time.perf_counter() - latency_start
        ) * 1000 / 20

    # Grafik akurasi
    plt.figure()
    plt.plot(
        [item["epoch"] for item in history],
        [item["train_accuracy"] * 100 for item in history],
        label="Train Accuracy"
    )
    plt.plot(
        [item["epoch"] for item in history],
        [item["val_accuracy"] * 100 for item in history],
        label="Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"ResNet-18 - {mode}")
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULT_DIR / f"accuracy_{mode}.png")
    plt.close()

    with open(
        RESULT_DIR / f"history_{mode}.csv",
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"Mode: {mode}")
    print(f"Akurasi validasi terbaik: {best_accuracy * 100:.2f}%")
    print(f"Epoch terbaik: {best_epoch}")
    print(f"Waktu training: {training_time / 60:.2f} menit")
    print(f"Latensi rata-rata: {latency_ms:.2f} ms")
    print(f"Model tersimpan: {model_path}")

    return {
        "mode": mode,
        "best_val_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "training_time_seconds": training_time,
        "latency_ms": latency_ms,
    }


# =========================
# MENJALANKAN TIGA MODE
# =========================
if __name__ == "__main__":
    results = []

    for mode in ["feature", "partial", "scratch"]:
        results.append(run_training(mode))

    with open(
        RESULT_DIR / "comparison.csv",
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mode",
                "best_val_accuracy",
                "best_epoch",
                "training_time_seconds",
                "latency_ms",
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSEMUA PELATIHAN SELESAI!")
    print("Hasil tersimpan di folder results.")