
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

print("XPU available:", torch.xpu.is_available())

if torch.xpu.is_available():
    print("GPU:", torch.xpu.get_device_name(0))

from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor
)

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"

print("\nLoading VideoMAE processor...")

processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)

print("Processor loaded!")
print("Image size:", processor.size)
print("Mean:", processor.image_mean)
print("Std:", processor.image_std)

print("\nLoading VideoMAE model...")

model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)

print("VideoMAE loaded successfully!")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Original classes:", model.config.num_labels)
