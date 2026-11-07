
import torch

from torch.utils.data import DataLoader
from config import DEVICE, BATCH_SIZE
from dataset import ViolenceDataset


train_dataset = ViolenceDataset(split="train")
val_dataset = ViolenceDataset(split="val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

videos, labels = next(iter(train_loader))

print("\nData Loader Test")
print("-------------------------")
print("Video batch shape:", videos.shape)
print("Labels:", labels)
print("Labels shape:", labels.shape)

# Transfer batch to Intel GPU
videos = videos.to(DEVICE)
labels = labels.to(DEVICE)

if DEVICE.type == "xpu":
    torch.xpu.synchronize()

print("\nGPU Transfer Successful")
print("Video device:", videos.device)
print("Label device:", labels.device)
print("Video data type:", videos.dtype)
