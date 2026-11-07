import re
import string
import joblib
import spacy
from pathlib import Path

# ==========================================================
# LOAD MODEL
# ==========================================================

project_root = Path(__file__).resolve().parent.parent

models_dir = project_root / "models"

model = joblib.load(models_dir / "incident_classifier.pkl")

vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")

label_encoder = joblib.load(models_dir / "label_encoder.pkl")

nlp = spacy.load("en_core_web_sm")

# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    doc = nlp(text)

    words = []

    for token in doc:

        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
        ):

            words.append(token.lemma_)

    return " ".join(words)

# ==========================================================
# PREDICT
# ==========================================================

while True:

    report = input("\nEnter Incident Report:\n")

    if report.lower() == "exit":
        break

    cleaned = clean_text(report)

    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)

    confidence = model.predict_proba(features).max()

    label = label_encoder.inverse_transform(prediction)[0]

    print("\nPrediction")

    print("----------------------")

    print("Incident :", label)

    print(f"Confidence : {confidence:.2%}")