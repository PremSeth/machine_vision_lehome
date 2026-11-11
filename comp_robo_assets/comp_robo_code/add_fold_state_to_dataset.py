import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from lerobot.datasets.lerobot_dataset import LeRobotDataset


#this finds the main lehome challenge folder  by taking the current path and moving 3 parent folders up
root = Path(__file__).resolve().parents[2]
source = root / "Datasets/example/top_long_merged" #this is the original ACT training dataset
output = root / "Datasets/example/top_long_with_fold_state" #this is where the new version of the dataset will go
device = "cuda" #we want to use the gpu to run the resnet over all the images quickly

#this loads the resnet weights and the order of its four labels
checkpoint = torch.load(
    root / "comp_robo_assets/resnet_training/fold_resnet.pt", map_location="cpu"
)
model = models.resnet18(weights=None) #we use none because our trained weights get loaded below
model.fc = nn.Linear(model.fc.in_features, 4) #this rebuilds the same four output head we trained with
model.load_state_dict(checkpoint["model"]) #this puts our trained weights into the resnet
model.to(device).eval() #this moves it to the gpu and puts it in inference mode

#this loads the original ACT dataset
dataset = LeRobotDataset(repo_id="repo_act", root=source)

#we only need the top camera video for the resnet so this removes the other videos from what gets loaded
dataset.meta.info["features"] = {
    key: value
    for key, value in dataset.meta.features.items()
    if value["dtype"] != "video" or key == "observation.images.top_rgb"
}
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8) #loads 64 frames at a time in their original order

#these are the same image normalization values that were used while training the resnet
#view changes them into the right shape to apply one value to each RGB channel in every image
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
probabilities = [] #this will store the four predicted values for every frame

#no_grad means we are only running inference and are not training or changing the resnet here
with torch.no_grad():
    for batch in loader:
        images = batch["observation.images.top_rgb"].to(device) #gets the top camera images from this batch
        images = F.interpolate(images, (224, 224), mode="bilinear", align_corners=False) #resizes them for resnet

        #this normalizes the images, runs the resnet, and turns its four outputs into probabilities
        probabilities.append(torch.softmax(model((images - mean) / std), dim=1).cpu())

probabilities = torch.cat(probabilities) #joins all the separate batches back into one ordered list

#this creates a separate dataset so we do not change the original top_long_merged data
output.mkdir(parents=True)
shutil.copytree(source / "data", output / "data") #copies the parquet training data
shutil.copytree(source / "meta", output / "meta") #copies the information describing the dataset

#the videos are large so this points to the originals instead of making another copy of all of them
(output / "videos").symlink_to(source / "videos", target_is_directory=True)

#this opens the parquet file where the values used to train ACT are stored
parquet_path = output / "data/chunk-000/file-000.parquet"
table = pq.read_table(parquet_path)

#this turns the resnet outputs into a parquet column where each row has four values
column = pa.array(probabilities.tolist(), type=pa.list_(pa.float32()))

#this adds that column into the dataset with the environment state name ACT will look for
pq.write_table(table.append_column("observation.environment_state", column), parquet_path)

#this opens the dataset information file so we can also tell it about the new column
info_path = output / "meta/info.json"
info = json.loads(info_path.read_text())

#this says the new input has four decimal values and saves what each value represents
info["features"]["observation.environment_state"] = {
    "dtype": "float32",
    "shape": [4],
    "names": checkpoint["classes"],
}
info_path.write_text(json.dumps(info, indent=4)) #saves the updated dataset information

print(output) #prints where the completed new dataset was saved
