import spacy
from pathlib import Path

# ==========================================================
# LOAD SPACY MODEL
# ==========================================================

print("=" * 60)
print("Loading spaCy Model...")
print("=" * 60)

nlp = spacy.load("en_core_web_sm")

print("spaCy Loaded Successfully!")

# ==========================================================
# ENTITY EXTRACTION FUNCTION
# ==========================================================
def extract_entities(report):

    doc = nlp(report)

    entities = {
        "Location": [],
        "Date": [],
        "Time": [],
        "Person": [],
        "Organization": [],
        "Keywords": []
    }

    # Campus Locations
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

    # Rule-based location extraction
    for location in campus_locations:
        if location.lower() in report.lower():
            entities["Location"].append(location)

    # spaCy entity extraction
    for ent in doc.ents:
        text = ent.text
        if text in campus_locations:
            continue

        if ent.label_ == "DATE":
            entities["Date"].append(text)

        elif ent.label_ == "TIME":
            entities["Time"].append(text)

        elif ent.label_ == "PERSON":
            entities["Person"].append(text)

        elif ent.label_ == "ORG":
            entities["Organization"].append(text)

    # Important keywords
    for token in doc:

        if (
            not token.is_stop
            and not token.is_punct
            and token.pos_ in ["NOUN", "PROPN", "VERB"]
        ):

            entities["Keywords"].append(token.lemma_.lower())

    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities

# ==========================================================
# TEST
# ==========================================================

while True:

    print("\nEnter Incident Report:")

    report = input()

    if report.lower() == "exit":
        break

    result = extract_entities(report)

    print("\nExtracted Entities")
    print("=" * 60)

    for key, value in result.items():

        print(f"{key:12} : {value}")