from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


root = Path(__file__).resolve().parents[2]
data_folder = root / "comp_robo_assets/resnet_training"
device = "cuda" if torch.cuda.is_available() else "cpu"

clean_images = transforms.Compose([
    transforms.Lambda(lambda image: image.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(data_folder, transform=clean_images)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

for epoch in range(10):
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")

torch.save(
    {"model": model.state_dict(), "classes": dataset.classes},
    data_folder / "fold_resnet.pt",
)
