
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from config import DEVICE, NUM_CLASSES, MODEL_PATH, OUTPUT_PATH
from dataset import ViolenceDataset


# 1. Load best checkpoint
checkpoint_path = MODEL_PATH / "r3d18_best.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

model = r3d_18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print("Loaded best checkpoint")
print("Best epoch:", checkpoint["epoch"])
print("Saved best validation F1:", checkpoint["best_val_f1"])


# 2. Load validation dataset
dataset = ViolenceDataset(split="val")

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)


# 3. Run evaluation
all_labels = []
all_predictions = []
all_probabilities = []

with torch.inference_mode():

    for videos, labels in loader:

        videos = videos.to(DEVICE)

        outputs = model(videos)

        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

if DEVICE.type == "xpu":
    torch.xpu.synchronize()


# 4. Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(
    all_labels, all_predictions, zero_division=0
)
recall = recall_score(
    all_labels, all_predictions, zero_division=0
)
f1 = f1_score(
    all_labels, all_predictions, zero_division=0
)

auc = roc_auc_score(
    all_labels,
    np.array(all_probabilities)[:, 1]
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)

print("\nBEST CHECKPOINT EVALUATION")
print("=" * 45)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {auc:.4f}")

print("\nConfusion Matrix")
print("Rows = Actual, Columns = Predicted")
print(cm)

print("\nClassification Report")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["NonFight", "Fight"],
        zero_division=0
    )
)


# 5. Save evaluation results
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

results = pd.DataFrame({
    "actual": all_labels,
    "predicted": all_predictions,
    "nonfight_probability": np.array(all_probabilities)[:, 0],
    "fight_probability": np.array(all_probabilities)[:, 1]
})

results.to_csv(
    OUTPUT_PATH / "r3d18_best_predictions.csv",
    index=False
)

metrics = pd.DataFrame([{
    "checkpoint_epoch": checkpoint["epoch"],
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": auc
}])

metrics.to_csv(
    OUTPUT_PATH / "r3d18_best_metrics.csv",
    index=False
)

print("\nEvaluation files saved in outputs/")
