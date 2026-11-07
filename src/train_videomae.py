
import torch
import torch.nn as nn
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from config import DEVICE, MODEL_PATH, OUTPUT_PATH
from videomae_dataset import VideoMAEDataset


MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"

EPOCHS = 2
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
MAX_TRAIN_SAMPLES = 400

MODEL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("\nI-MAGE VIDEOMAE TRAINING")
print("=" * 45)
print("Device:", DEVICE)

# 1. Dataset
train_dataset = VideoMAEDataset(
    split="train",
    max_samples=MAX_TRAIN_SAMPLES
)

val_dataset = VideoMAEDataset(
    split="val"
)

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

# 2. Load pretrained VideoMAE
print("\nLoading pretrained VideoMAE...")

model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    ignore_mismatched_sizes=True,
    label2id={"NonFight": 0, "Fight": 1},
    id2label={0: "NonFight", 1: "Fight"}
)

model = model.to(DEVICE)

# 3. Freeze backbone, train classification head
model.videomae.requires_grad_(False)

for parameter in model.classifier.parameters():
    parameter.requires_grad = True

trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", trainable)

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

criterion = nn.CrossEntropyLoss()

best_f1 = -1
history = []

# 4. Training
for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    model.train()

    # Keep frozen transformer in evaluation mode
    model.videomae.eval()

    train_losses = []
    train_labels = []
    train_predictions = []

    for videos, labels in tqdm(
        train_loader, desc="Training"
    ):

        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(pixel_values=videos)
        logits = outputs.logits

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        train_predictions.extend(
            logits.argmax(dim=1).detach().cpu().tolist()
        )

        train_labels.extend(labels.cpu().tolist())

    if DEVICE.type == "xpu":
        torch.xpu.synchronize()

    # 5. Validation
    model.eval()

    val_losses = []
    val_labels = []
    val_predictions = []
    val_probabilities = []

    with torch.inference_mode():

        for videos, labels in tqdm(
            val_loader, desc="Validation"
        ):

            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(pixel_values=videos).logits

            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)

            val_losses.append(loss.item())

            val_predictions.extend(
                logits.argmax(dim=1).cpu().tolist()
            )

            val_labels.extend(labels.cpu().tolist())

            val_probabilities.extend(
                probabilities.cpu().tolist()
            )

    if DEVICE.type == "xpu":
        torch.xpu.synchronize()

    # 6. Metrics
    train_loss = sum(train_losses) / len(train_losses)

    train_accuracy = accuracy_score(
        train_labels, train_predictions
    )

    val_loss = sum(val_losses) / len(val_losses)

    accuracy = accuracy_score(
        val_labels, val_predictions
    )

    precision = precision_score(
        val_labels, val_predictions, zero_division=0
    )

    recall = recall_score(
        val_labels, val_predictions, zero_division=0
    )

    f1 = f1_score(
        val_labels, val_predictions, zero_division=0
    )

    auc = roc_auc_score(
        val_labels,
        [p[1] for p in val_probabilities]
    )

    print("\nEpoch results")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })

    # 7. Save best model
    if f1 > best_f1:

        best_f1 = f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_f1": best_f1,
                "model_id": MODEL_ID,
                "class_names": ["NonFight", "Fight"],
                "num_frames": 16,
                "image_size": 224,
                "backbone_frozen": True
            },
            MODEL_PATH / "videomae_best.pth"
        )

        print("Best VideoMAE checkpoint saved!")

# 8. Save history
pd.DataFrame(history).to_csv(
    OUTPUT_PATH / "videomae_training_history.csv",
    index=False
)

# 9. Save final epoch confusion matrix
cm = confusion_matrix(
    val_labels, val_predictions, labels=[0, 1]
)

print("\nFinal epoch confusion matrix:")
print(cm)

print("\nBest VideoMAE F1:", best_f1)
print("VideoMAE training completed.")
