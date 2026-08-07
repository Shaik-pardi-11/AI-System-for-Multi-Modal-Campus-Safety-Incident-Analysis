import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==========================================================
# PROJECT PATHS
# ==========================================================

project_root = Path(__file__).resolve().parent.parent

models_dir = project_root / "models"

# ==========================================================
# LOAD FEATURES
# ==========================================================

print("=" * 60)
print("Loading Feature Files...")
print("=" * 60)

X = joblib.load(models_dir / "X_features.pkl")
y = joblib.load(models_dir / "y_labels.pkl")
label_encoder = joblib.load(models_dir / "label_encoder.pkl")

print(f"Feature Matrix : {X.shape}")
print(f"Labels         : {len(y)}")

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

# ==========================================================
# MODEL
# ==========================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training Completed!")

# ==========================================================
# PREDICTIONS
# ==========================================================

print("\nPredicting Test Data...")

y_pred = model.predict(X_test)

# ==========================================================
# EVALUATION
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    models_dir / "incident_classifier.pkl"
)

print("\nModel Saved Successfully!")

print(models_dir / "incident_classifier.pkl")

print("\nDone!")