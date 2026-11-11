from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# This line below finds the path to the current file, then goes up three parent folders
root = Path(__file__).resolve().parents[2]

data_folder = root / "comp_robo_assets/resnet_training" #this finds the absolute path for the data folder
device = "cuda" if torch.cuda.is_available() else "cpu" # this allocates traning onto the gpu if there else cpi

# This makes a sequence of transformations.
clean_images = transforms.Compose([
    transforms.Lambda(lambda image: image.convert("RGB")), #makes sure its RGB
    transforms.Resize((224, 224)), #makes sure all the same size
    transforms.ToTensor(), #Maps all values from 0 to 1 so weights can learn without exploding values
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ), #this maps color values accoding to the same distribution of values resnet was trained on
])

# Use each folder name as its image label and use a data loader to load the images in batches.
# ImageFolder orders the labels alphabetically.
dataset = datasets.ImageFolder(data_folder, transform=clean_images)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

# Load a pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#In the readme I touch on using a CNN trained from scratch and in order to that all we would need to do 
# is change weights = None

#this changes the output head of the model to match our number of labels 
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

# Set up the loss and optimizer used for training.
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# Train the whole network for ten passes over the screenshot dataset.
for epoch in range(10):
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")

# Save the trained weights and label order so they can be used later.
torch.save(
    {"model": model.state_dict(), "classes": dataset.classes},
    data_folder / "fold_resnet.pt",
)
