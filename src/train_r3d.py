
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

from config import (
    DEVICE,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    OUTPUT_PATH
)

from dataset import ViolenceDataset


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("\nI-MAGE R3D-18 TRAINING")
print("=" * 50)
print("Device:", DEVICE)

if DEVICE.type == "xpu":
    print("GPU:", torch.xpu.get_device_name(0))

print("Batch size:", BATCH_SIZE)
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

train_dataset = ViolenceDataset(split="train")
val_dataset = ViolenceDataset(split="val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 3. Load pretrained R3D-18
# --------------------------------------------------

print("\nLoading pretrained R3D-18...")

weights = R3D_18_Weights.DEFAULT

model = r3d_18(weights=weights)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

best_f1 = -1.0
history = []


# --------------------------------------------------
# 4. Training and validation
# --------------------------------------------------

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    # Training
    model.train()

    train_losses = []
    train_predictions = []
    train_labels = []

    progress = tqdm(train_loader, desc="Training")

    for videos, labels in progress:

        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(videos)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        predictions = torch.argmax(outputs, dim=1)

        train_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        train_labels.extend(
            labels.detach().cpu().tolist()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    if DEVICE.type == "xpu":
        torch.xpu.synchronize()

    train_loss = sum(train_losses) / len(train_losses)

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )

    # Validation
    model.eval()

    val_losses = []
    val_predictions = []
    val_labels = []

    with torch.inference_mode():

        for videos, labels in tqdm(
            val_loader,
            desc="Validation"
        ):

            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(videos)

            loss = criterion(outputs, labels)

            val_losses.append(loss.item())

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_predictions.extend(
                predictions.cpu().tolist()
            )

            val_labels.extend(
                labels.cpu().tolist()
            )

    if DEVICE.type == "xpu":
        torch.xpu.synchronize()

    val_loss = sum(val_losses) / len(val_losses)

    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )

    val_precision = precision_score(
        val_labels,
        val_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        val_labels,
        val_predictions,
        zero_division=0
    )

    val_f1 = f1_score(
        val_labels,
        val_predictions,
        zero_division=0
    )

    # Display results
    print("\nEpoch Results")
    print("-" * 40)

    print(f"Train Loss:     {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")

    print(f"Val Loss:       {val_loss:.4f}")
    print(f"Val Accuracy:   {val_accuracy:.4f}")
    print(f"Val Precision:  {val_precision:.4f}")
    print(f"Val Recall:     {val_recall:.4f}")
    print(f"Val F1 Score:   {val_f1:.4f}")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1
    })

    # Save best model
    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_f1,
                "class_names": ["NonFight", "Fight"],
                "num_frames": 16,
                "image_size": 112
            },
            MODEL_PATH / "r3d18_best.pth"
        )

        print("\nBest model checkpoint saved!")


# --------------------------------------------------
# 5. Save training history
# --------------------------------------------------

history_df = pd.DataFrame(history)

history_df.to_csv(
    OUTPUT_PATH / "r3d18_training_history.csv",
    index=False
)

print("\nTraining history saved.")

# --------------------------------------------------
# 6. Final validation confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    val_labels,
    val_predictions,
    labels=[0, 1]
)

print("\nFinal Epoch Confusion Matrix")
print("Rows = Actual, Columns = Predicted")
print(cm)

print("\nI-Mage R3D-18 training completed.")
print("Best validation F1:", best_f1)
print("Model:", MODEL_PATH / "r3d18_best.pth")
