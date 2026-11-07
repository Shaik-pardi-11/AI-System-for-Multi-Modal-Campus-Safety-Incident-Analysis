import pandas as pd
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# PROJECT PATHS
# ==========================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "processed" / "cleaned_dataset.csv"

models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Cleaned Dataset...")
print("=" * 60)

df = pd.read_csv(input_file)

print(f"Total Records : {len(df)}")

# ==========================================================
# REMOVE EMPTY VALUES
# ==========================================================

df = df.dropna(subset=["Clean_Report", "Label"])

print(f"Records After Cleaning : {len(df)}")

# ==========================================================
# TEXT DATA
# ==========================================================

X_text = df["Clean_Report"].astype(str)

# ==========================================================
# LABELS
# ==========================================================

y = df["Label"].astype(str)

# ==========================================================
# TF-IDF VECTORIZATION
# ==========================================================

print("\nCreating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    lowercase=False
)

X = vectorizer.fit_transform(X_text)

print(f"TF-IDF Shape : {X.shape}")

# ==========================================================
# LABEL ENCODING
# ==========================================================

print("\nEncoding Labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print(f"Total Classes : {len(label_encoder.classes_)}")

print("\nClasses:")

for idx, label in enumerate(label_encoder.classes_):
    print(f"{idx:2d} -> {label}")

# ==========================================================
# SAVE OBJECTS
# ==========================================================

print("\nSaving Files...")

joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.pkl")

joblib.dump(label_encoder, models_dir / "label_encoder.pkl")

joblib.dump(X, models_dir / "X_features.pkl")

joblib.dump(y_encoded, models_dir / "y_labels.pkl")

# Save class names (optional but useful)

joblib.dump(label_encoder.classes_, models_dir / "classes.pkl")

# ==========================================================
# SHOW SAMPLE FEATURES
# ==========================================================

print("\nVocabulary Size :", len(vectorizer.vocabulary_))

feature_names = vectorizer.get_feature_names_out()

print("\nFirst 20 Features:")

for feature in feature_names[:20]:
    print(feature)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(f"Vocabulary Size      : {len(vectorizer.vocabulary_)}")
print(f"Feature Matrix Shape : {X.shape}")
print(f"Number of Classes    : {len(label_encoder.classes_)}")

print("\nSaved Files")

print(f"✔ {models_dir / 'tfidf_vectorizer.pkl'}")
print(f"✔ {models_dir / 'label_encoder.pkl'}")
print(f"✔ {models_dir / 'X_features.pkl'}")
print(f"✔ {models_dir / 'y_labels.pkl'}")
print(f"✔ {models_dir / 'classes.pkl'}")

print("\nFeature Engineering Completed Successfully!")