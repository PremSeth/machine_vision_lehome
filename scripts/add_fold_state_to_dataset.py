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


root = Path(__file__).resolve().parents[1]
source = root / "Datasets/example/top_long_merged"
output = root / "Datasets/example/top_long_with_fold_state"
device = "cuda"

checkpoint = torch.load(root / "resnet_training/fold_resnet.pt", map_location="cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(checkpoint["model"])
model.to(device).eval()

dataset = LeRobotDataset(repo_id="repo_act", root=source)
dataset.meta.info["features"] = {
    key: value
    for key, value in dataset.meta.features.items()
    if value["dtype"] != "video" or key == "observation.images.top_rgb"
}
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
probabilities = []

with torch.no_grad():
    for batch in loader:
        images = batch["observation.images.top_rgb"].to(device)
        images = F.interpolate(images, (224, 224), mode="bilinear", align_corners=False)
        probabilities.append(torch.softmax(model((images - mean) / std), dim=1).cpu())

probabilities = torch.cat(probabilities)

output.mkdir(parents=True)
shutil.copytree(source / "data", output / "data")
shutil.copytree(source / "meta", output / "meta")
(output / "videos").symlink_to(source / "videos", target_is_directory=True)

parquet_path = output / "data/chunk-000/file-000.parquet"
table = pq.read_table(parquet_path)
column = pa.array(probabilities.tolist(), type=pa.list_(pa.float32()))
pq.write_table(table.append_column("observation.environment_state", column), parquet_path)

info_path = output / "meta/info.json"
info = json.loads(info_path.read_text())
info["features"]["observation.environment_state"] = {
    "dtype": "float32",
    "shape": [4],
    "names": checkpoint["classes"],
}
info_path.write_text(json.dumps(info, indent=4))

print(output)
