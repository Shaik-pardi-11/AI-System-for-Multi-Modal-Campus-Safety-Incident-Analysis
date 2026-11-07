import pandas as pd
import re
import string
import os
from pathlib import Path

import nltk
from nltk.corpus import stopwords
import spacy

# ==========================================================
# DOWNLOAD REQUIRED NLTK DATA (Only downloads once)
# ==========================================================

nltk.download("stopwords")

# ==========================================================
# LOAD SPACY MODEL
# ==========================================================

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("\nERROR:")
    print("SpaCy model not installed.")
    print("Run this command:\n")
    print("python -m spacy download en_core_web_sm\n")
    raise

# ==========================================================
# LOAD STOPWORDS
# ==========================================================

stop_words = set(stopwords.words("english"))

# ==========================================================
# PROJECT PATHS
# ==========================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "processed" / "campus_incidents.csv"

output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "cleaned_dataset.csv"

# ==========================================================
# LOAD DATASET
# ==========================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove Email
    text = re.sub(r"\S+@\S+", "", text)

    # Remove Numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_and_save(df_in):
    df = df_in.copy()

    print("\nCleaning Reports...\n")

    if "Report" not in df.columns:
        print("\nCreating Report column...")
        df["Report"] = (
            df["Incident Description"].fillna("").astype(str)
            + " "
            + df["Incident Category"].fillna("").astype(str)
            + " at "
            + df["Location"].fillna("").astype(str)
        )

    # Optional quick sample run: set env var SAMPLE_ROWS to an integer to limit rows
    sample_rows = int(os.getenv("SAMPLE_ROWS", "0"))
    if sample_rows > 0:
        print(f"Limiting to first {sample_rows} rows for quick run")
        df = df.head(sample_rows).copy()

    # Prepare reports
    reports = df["Report"].fillna("").astype(str).apply(clean_text).tolist()

    # Batch / multiprocessing settings
    default_procs = max(1, (os.cpu_count() or 1) - 1)
    n_procs = int(os.getenv("N_PROCESSES", str(default_procs)))
    batch_size = int(os.getenv("BATCH_SIZE", "1000"))

    print(f"Processing reports with spaCy n_process={n_procs}, batch_size={batch_size}...")

    cleaned_reports = []
    # Use disable to skip expensive components
    pipe_kwargs = {"batch_size": batch_size, "disable": ["ner", "parser"]}
    if n_procs > 1:
        pipe_kwargs["n_process"] = n_procs

    idx = 0
    for doc in nlp.pipe(reports, **pipe_kwargs):
        words = [
            token.lemma_
            for token in doc
            if (
                token.text not in stop_words
                and not token.is_punct
                and not token.is_space
                and len(token.text) > 2
            )
        ]
        cleaned_reports.append(" ".join(words))
        idx += 1
        if idx % 10000 == 0:
            print(f"Processed {idx} reports...")

    df["Clean_Report"] = cleaned_reports

    # Remove empty rows
    df = df[df["Clean_Report"].str.strip() != ""]

    # Save
    df.to_csv(output_file, index=False)

    print("=" * 60)
    print("TEXT PREPROCESSING COMPLETED")
    print("=" * 60)

    print("\nSample Output:\n")
    print(df[["Report", "Clean_Report"]].head(10))

    print("\n")
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    print(f"Total Records        : {len(df)}")
    print(f"Total Columns        : {len(df.columns)}")

    print("\nColumns:")
    for col in df.columns:
        print(f" • {col}")

    print("\nSaved To:")
    print(output_file)

    print("\nDone Successfully!")


if __name__ == "__main__":
    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    df = pd.read_csv(input_file)

    print(f"\nDataset Loaded Successfully")
    print(f"Total Records : {len(df)}")

    process_and_save(df)