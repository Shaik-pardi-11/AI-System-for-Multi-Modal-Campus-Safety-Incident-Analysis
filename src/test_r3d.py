
import torch
import torch.nn as nn

from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights
)

from torch.utils.data import DataLoader

from config import DEVICE, NUM_CLASSES
from dataset import ViolenceDataset


print("\nLoading R3D-18 model...")

# Load pretrained Kinetics-400 weights
weights = R3D_18_Weights.DEFAULT

model = r3d_18(weights=weights)

# Replace original 400-class output layer
model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)
model.eval()

print("R3D-18 loaded successfully")
print("Device:", DEVICE)


# Load one actual video batch
dataset = ViolenceDataset(split="train")

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

videos, labels = next(iter(loader))

videos = videos.to(DEVICE)

print("\nInput shape:", videos.shape)

# GPU inference test
try:
    with torch.inference_mode():
        outputs = model(videos)

    if DEVICE.type == "xpu":
        torch.xpu.synchronize()

    probabilities = torch.softmax(outputs, dim=1)

    predicted_class = torch.argmax(
        probabilities, dim=1
    ).item()

    print("\nR3D-18 GPU TEST SUCCESSFUL")
    print("Output shape:", outputs.shape)
    print("Predicted class:", predicted_class)
    print("Probabilities:", probabilities.cpu().numpy())

    if predicted_class == 0:
        print("Prediction: NonFight")
    else:
        print("Prediction: Fight")

except Exception as error:
    print("\nR3D-18 GPU TEST FAILED")
    print(type(error).__name__, error)
    raise
