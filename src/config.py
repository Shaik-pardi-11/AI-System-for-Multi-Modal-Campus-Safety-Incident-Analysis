
from pathlib import Path
import torch

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "RWF-2000"

TRAIN_DIR = DATASET_PATH / "train"
VAL_DIR = DATASET_PATH / "val"

TRAIN_FIGHT = TRAIN_DIR / "Train_Fight"
TRAIN_NONFIGHT = TRAIN_DIR / "Train_NonFight"

VAL_FIGHT = VAL_DIR / "Val_Fight"
VAL_NONFIGHT = VAL_DIR / "Val_NonFight"

MODEL_PATH = PROJECT_ROOT / "models"
OUTPUT_PATH = PROJECT_ROOT / "outputs"
LOG_PATH = PROJECT_ROOT / "logs"

# Device selection
if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Class mapping
CLASS_TO_IDX = {
    "NonFight": 0,
    "Fight": 1
}

CLASS_NAMES = ["NonFight", "Fight"]

# Video configuration
NUM_FRAMES = 16
IMAGE_SIZE = 112
NUM_CLASSES = 2

# Initial training configuration
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 0.0001

print("I-Mage Configuration Loaded")
print("Device:", DEVICE)

if DEVICE.type == "xpu":
    print("GPU:", torch.xpu.get_device_name(0))
