
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from videomae_dataset import VideoMAEDataset

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"

CHECKPOINT = ROOT / "models" / "videomae_best.pth"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["NonFight", "Fight"]

BATCH_SIZE = 1

if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("I-MAGE - VIDEOMAE MODEL EVALUATION")
print("=" * 60)
print("Device:", DEVICE)


# =========================================================
# LOAD VALIDATION DATASET
# =========================================================

print("\nLoading validation dataset...")

val_dataset = VideoMAEDataset(split="val")

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print("Validation videos:", len(val_dataset))


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

print("\nLoading trained VideoMAE checkpoint...")

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )

model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    ignore_mismatched_sizes=True,
    label2id={"NonFight": 0, "Fight": 1},
    id2label={0: "NonFight", 1: "Fight"}
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

state_dict = checkpoint.get(
    "model_state_dict",
    checkpoint
)

model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

print("Checkpoint loaded successfully.")


# =========================================================
# RUN INFERENCE
# =========================================================

all_labels = []
all_predictions = []
all_probabilities = []
prediction_rows = []

print("\nStarting evaluation...\n")

with torch.inference_mode():

    for index, (videos, labels) in enumerate(val_loader):

        # Dataset returns [B, T, C, H, W]
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        assert videos.ndim == 5
        assert videos.shape[2] == 3, (
            f"Incorrect VideoMAE input shape: {videos.shape}"
        )

        outputs = model(
            pixel_values=videos
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        actual = labels.cpu().numpy()
        predicted = predictions.cpu().numpy()
        probs = probabilities.cpu().numpy()

        all_labels.extend(actual.tolist())
        all_predictions.extend(predicted.tolist())
        all_probabilities.extend(probs[:, 1].tolist())

        for i in range(len(actual)):

            prediction_rows.append({
                "video_index": index + i,
                "actual": CLASS_NAMES[int(actual[i])],
                "predicted": CLASS_NAMES[int(predicted[i])],
                "fight_probability": float(probs[i][1]),
                "nonfight_probability": float(probs[i][0]),
                "correct": bool(actual[i] == predicted[i])
            })

        if (index + 1) % 25 == 0 or (index + 1) == len(val_loader):
            print(
                f"Processed {index + 1}/{len(val_loader)} videos"
            )


# =========================================================
# CALCULATE METRICS
# =========================================================

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

try:
    auc = roc_auc_score(y_true, y_prob)
except ValueError:
    auc = float("nan")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 60)
print("VIDEOMAE EVALUATION RESULTS")
print("=" * 60)

print(f"Total videos : {len(y_true)}")
print(f"Accuracy     : {accuracy * 100:.2f}%")
print(f"Precision    : {precision * 100:.2f}%")
print(f"Recall       : {recall * 100:.2f}%")
print(f"F1 Score     : {f1 * 100:.2f}%")
print(f"ROC-AUC      : {auc * 100:.2f}%")

print("\nConfusion Matrix")
print("Rows = Actual | Columns = Predicted")
print("Order: NonFight, Fight")
print(cm)

print("\nClassification Report")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# =========================================================
# SAVE RESULTS
# =========================================================

predictions_df = pd.DataFrame(prediction_rows)

predictions_path = (
    OUTPUT_DIR / "videomae_validation_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

metrics_df = pd.DataFrame([{
    "model": "VideoMAE",
    "total_videos": len(y_true),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": auc
}])

metrics_path = (
    OUTPUT_DIR / "videomae_validation_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

confusion_df = pd.DataFrame(
    cm,
    index=["Actual_NonFight", "Actual_Fight"],
    columns=["Predicted_NonFight", "Predicted_Fight"]
)

confusion_path = (
    OUTPUT_DIR / "videomae_confusion_matrix.csv"
)

confusion_df.to_csv(confusion_path)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    zero_division=0
)

report_path = OUTPUT_DIR / "videomae_classification_report.txt"
report_path.write_text(report, encoding="utf-8")

print("\n" + "=" * 60)
print("EVALUATION FILES SAVED")
print("=" * 60)

print(predictions_path)
print(metrics_path)
print(confusion_path)
print(report_path)

print("\nVideoMAE evaluation completed successfully.")
