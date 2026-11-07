from fastapi import FastAPI
from pydantic import BaseModel

import json
import re
import string
import joblib
import spacy

from pathlib import Path

# ==========================================================
# LOAD MODELS
# ==========================================================

project_root = Path(__file__).resolve().parent.parent

models_dir = project_root / "models"

model = joblib.load(models_dir / "incident_classifier.pkl")
vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
label_encoder = joblib.load(models_dir / "label_encoder.pkl")

nlp = spacy.load("en_core_web_sm")

# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(
    title="Campus Safety NLP API",
    description="Incident Classification API",
    version="1.0"
)

# ==========================================================
# REQUEST MODEL
# ==========================================================

class IncidentRequest(BaseModel):
    report: str

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
# ENTITY EXTRACTION
# ==========================================================

campus_locations = [
    "Library",
    "Hostel A",
    "Hostel B",
    "Playground",
    "Parking Area",
    "Main Gate",
    "Engineering Block",
    "Computer Lab",
    "Administration Block",
    "Cafeteria"
]

def extract_entities(report):

    doc = nlp(report)

    entities = {
        "location": [],
        "date": [],
        "time": [],
        "person": [],
        "organization": [],
        "keywords": []
    }

    for location in campus_locations:
        if location.lower() in report.lower():
            entities["location"].append(location)

    for ent in doc.ents:

        if ent.text in campus_locations:
            continue

        if ent.label_ == "DATE":
            entities["date"].append(ent.text)

        elif ent.label_ == "TIME":
            entities["time"].append(ent.text)

        elif ent.label_ == "PERSON":
            entities["person"].append(ent.text)

        elif ent.label_ == "ORG":
            entities["organization"].append(ent.text)

    for token in doc:

        if (
            not token.is_stop
            and not token.is_punct
            and token.pos_ in ["NOUN", "PROPN", "VERB"]
        ):
            entities["keywords"].append(token.lemma_.lower())

    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities

# ==========================================================
# HOME
# ==========================================================

@app.get("/")
def home():

    return {
        "message": "Campus Safety NLP API is Running"
    }

# ==========================================================
# PREDICT
# ==========================================================

@app.post("/predict")
def predict(request: IncidentRequest):

    cleaned = clean_text(request.report)

    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)

    confidence = float(model.predict_proba(features).max())

    incident = label_encoder.inverse_transform(prediction)[0]

    entities = extract_entities(request.report)

    return {
        "incident_type": incident,
        "confidence": round(confidence, 4),
        "entities": entities
    }