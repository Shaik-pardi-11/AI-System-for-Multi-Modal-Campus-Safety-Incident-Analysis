
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

R3D_FILE = OUTPUT / "r3d18_best_metrics.csv"
VIDEOMAE_FILE = OUTPUT / "videomae_validation_metrics.csv"

# Rows = actual [NonFight, Fight]
# Columns = predicted [NonFight, Fight]
R3D_CM = np.array([
    [173, 27],
    [28, 172]
])

VIDEOMAE_CM = np.array([
    [161, 39],
    [14, 186]
])

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc"
]


# =========================================================
# LOAD METRICS
# =========================================================

print("=" * 60)
print("I-MAGE - MODEL COMPARISON")
print("=" * 60)

if not R3D_FILE.exists():
    raise FileNotFoundError(R3D_FILE)

if not VIDEOMAE_FILE.exists():
    raise FileNotFoundError(VIDEOMAE_FILE)

r3d = pd.read_csv(R3D_FILE)
videomae = pd.read_csv(VIDEOMAE_FILE)

r3d_row = r3d.iloc[0]
video_row = videomae.iloc[0]

comparison = pd.DataFrame({
    "Metric": METRICS,
    "R3D-18": [float(r3d_row[m]) for m in METRICS],
    "VideoMAE": [float(video_row[m]) for m in METRICS]
})

comparison["R3D-18 (%)"] = comparison["R3D-18"] * 100
comparison["VideoMAE (%)"] = comparison["VideoMAE"] * 100
comparison["Difference (percentage points)"] = (
    comparison["VideoMAE (%)"] - comparison["R3D-18 (%)"]
)

comparison.to_csv(
    OUTPUT / "model_comparison.csv",
    index=False
)


# =========================================================
# PRINT COMPARISON
# =========================================================

print("\nMODEL PERFORMANCE COMPARISON")
print("-" * 60)

for _, row in comparison.iterrows():
    print(
        f"{row['Metric']:12s} | "
        f"R3D-18: {row['R3D-18 (%)']:.2f}% | "
        f"VideoMAE: {row['VideoMAE (%)']:.2f}%"
    )


# =========================================================
# PERFORMANCE COMPARISON CHART
# =========================================================

x = np.arange(len(METRICS))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(
    x - width / 2,
    comparison["R3D-18 (%)"],
    width,
    label="R3D-18"
)

bars2 = ax.bar(
    x + width / 2,
    comparison["VideoMAE (%)"],
    width,
    label="VideoMAE"
)

ax.set_ylabel("Score (%)")
ax.set_title("I-Mage: R3D-18 vs VideoMAE Performance")
ax.set_xticks(x)
ax.set_xticklabels(
    ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis="y", alpha=0.25)

ax.bar_label(bars1, fmt="%.1f%%", padding=3)
ax.bar_label(bars2, fmt="%.1f%%", padding=3)

plt.tight_layout()
plt.savefig(
    OUTPUT / "model_performance_comparison.png",
    dpi=300
)
plt.close()


# =========================================================
# CONFUSION MATRIX VISUALIZATION
# =========================================================

def plot_confusion_matrix(matrix, title, filename):

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        matrix,
        interpolation="nearest"
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["NonFight", "Fight"])
    ax.set_yticklabels(["NonFight", "Fight"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    fig.colorbar(image, ax=ax)
    plt.tight_layout()

    plt.savefig(
        OUTPUT / filename,
        dpi=300
    )

    plt.close()


plot_confusion_matrix(
    R3D_CM,
    "R3D-18 Confusion Matrix",
    "r3d18_confusion_matrix.png"
)

plot_confusion_matrix(
    VIDEOMAE_CM,
    "VideoMAE Confusion Matrix",
    "videomae_confusion_matrix.png"
)


# =========================================================
# SIDE-BY-SIDE CONFUSION MATRIX
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

matrices = [
    (R3D_CM, "R3D-18"),
    (VIDEOMAE_CM, "VideoMAE")
]

for ax, (matrix, title) in zip(axes, matrices):

    image = ax.imshow(matrix)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["NonFight", "Fight"])
    ax.set_yticklabels(["NonFight", "Fight"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

fig.colorbar(image, ax=axes)
fig.suptitle("I-Mage Model Confusion Matrix Comparison")

plt.tight_layout()

plt.savefig(
    OUTPUT / "confusion_matrix_comparison.png",
    dpi=300
)

plt.close()


# =========================================================
# GENERATE REVIEW REPORT
# =========================================================

best_accuracy = comparison.loc[
    comparison["R3D-18 (%)"].idxmax(), "Metric"
]

report = f"""
I-MAGE
INTELLIGENT VIOLENCE AND NON-VIOLENCE DETECTION SYSTEM

MODEL EVALUATION AND COMPARISON REPORT
============================================================

1. EXPERIMENT OVERVIEW

Dataset: RWF-2000
Validation videos: 400
Classes: NonFight, Fight
Hardware: Intel Arc Graphics
Execution device: PyTorch XPU

2. MODEL PERFORMANCE

{comparison.to_string(index=False)}

3. R3D-18 CONFUSION MATRIX

{R3D_CM}

True Negatives: {R3D_CM[0, 0]}
False Positives: {R3D_CM[0, 1]}
False Negatives: {R3D_CM[1, 0]}
True Positives: {R3D_CM[1, 1]}

4. VIDEOMAE CONFUSION MATRIX

{VIDEOMAE_CM}

True Negatives: {VIDEOMAE_CM[0, 0]}
False Positives: {VIDEOMAE_CM[0, 1]}
False Negatives: {VIDEOMAE_CM[1, 0]}
True Positives: {VIDEOMAE_CM[1, 1]}

5. OBSERVATIONS

VideoMAE achieved higher validation recall and F1-score.
R3D-18 achieved higher validation precision.

VideoMAE detected more Fight videos but also produced
more false positive predictions.

6. EXPERIMENTAL LIMITATIONS

The two models used different training configurations.

R3D-18:
- Full model fine-tuning.
- 1,600 training videos.
- Best checkpoint from epoch 2.

VideoMAE:
- Frozen pretrained backbone.
- Classifier-head training.
- 400 balanced training videos.

Therefore, this comparison is not a controlled architecture
benchmark.

The validation set was also used for checkpoint selection.
These results should not be described as independent test
performance.

7. CONCLUSION

Both models demonstrate promising validation performance
for binary Fight and NonFight video classification.

VideoMAE provides higher Fight recall, while R3D-18
provides higher precision in the current experiments.

Further testing on unseen campus CCTV footage is required
before real-world deployment.

============================================================
END OF REPORT
"""

report_path = OUTPUT / "I-Mage_Model_Evaluation_Report.txt"

report_path.write_text(
    report,
    encoding="utf-8"
)


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("COMPARISON COMPLETED")
print("=" * 60)

files = [
    "model_comparison.csv",
    "model_performance_comparison.png",
    "r3d18_confusion_matrix.png",
    "videomae_confusion_matrix.png",
    "confusion_matrix_comparison.png",
    "I-Mage_Model_Evaluation_Report.txt"
]

for filename in files:
    print("Generated:", OUTPUT / filename)

print("\nAll comparison files generated successfully.")
