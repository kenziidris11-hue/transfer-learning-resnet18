
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

MODEL_PATH = "results/resnet18_partial.pth"
IMAGE_PATH = "dataset_raw/mur"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

path = Path(IMAGE_PATH)

if path.is_dir():
    images = [
        f for f in path.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]
    if not images:
        raise FileNotFoundError("Tidak ada gambar di folder tersebut.")
    path = images[0]

image = Image.open(path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print("\n=== HASIL PREDIKSI ===")
print("Gambar:", path)
print("Prediksi:", class_names[predicted.item()])
print(f"Keyakinan model: {confidence.item() * 100:.2f}%")