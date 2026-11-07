import pandas as pd
from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "raw" / "Police_Department_Incident_Reports__2018_to_Present_20260725.csv"

output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "campus_incidents.csv"

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(input_file)

# ==========================================================
# KEEP REQUIRED COLUMNS
# ==========================================================

df = df[
    [
        "Incident Description",
        "Incident Category",
        "Incident Date",
        "Incident Time",
        "Police District"
    ]
]

# ==========================================================
# MAP INCIDENT CATEGORIES TO CAMPUS LABELS
# ==========================================================
category_mapping = {

    # ----------------------------
    # Fight
    # ----------------------------
    "Assault": "Fight",
    "Battery": "Fight",

    # ----------------------------
    # Theft
    # ----------------------------
    "Robbery": "Theft",
    "Burglary": "Theft",
    "Larceny Theft": "Theft",
    "Motor Vehicle Theft": "Theft",
    "Motor Vehicle Theft?": "Theft",
    "Lost Property": "Theft",
    "Recovered Vehicle": "Theft",
    "Stolen Property": "Theft",
    "Vehicle Misplaced": "Theft",

    # ----------------------------
    # Fire
    # ----------------------------
    "Fire Report": "Fire",
    "Arson": "Fire",

    # ----------------------------
    # Medical Emergency
    # ----------------------------
    "Missing Person": "Medical Emergency",
    "Suicide": "Medical Emergency",

    # ----------------------------
    # Suspicious Activity
    # ----------------------------
    "Suspicious Occ": "Suspicious Activity",
    "Suspicious": "Suspicious Activity",

    # ----------------------------
    # Unauthorized Access
    # ----------------------------
    "Trespass": "Unauthorized Access",
    "Warrant": "Unauthorized Access",

    # ----------------------------
    # Vandalism
    # ----------------------------
    "Vandalism": "Vandalism",
    "Malicious Mischief": "Vandalism",

    # ----------------------------
    # Fraud
    # ----------------------------
    "Fraud": "Fraud",
    "Forgery And Counterfeiting": "Fraud",
    "Embezzlement": "Fraud",

    # ----------------------------
    # Weapons
    # ----------------------------
    "Weapons Offense": "Weapons",
    "Weapons Carrying Etc": "Weapons",
    "Weapons Offence": "Weapons",

    # ----------------------------
    # Everything Else
    # ----------------------------
    "Drug Offense": "Other",
    "Drug Violation": "Other",
    "Traffic Collision": "Other",
    "Traffic Violation Arrest": "Other",
    "Vehicle Impounded": "Other",
    "Courtesy Report": "Other",
    "Case Closure": "Other",
    "Civil Sidewalks": "Other",
    "Disorderly Conduct": "Other",
    "Gambling": "Other",
    "Liquor Laws": "Other",
    "Non-Criminal": "Other",
    "Other": "Other",
    "Other Miscellaneous": "Other",
    "Other Offenses": "Other",
    "Miscellaneous Investigation": "Other",
    "Offences Against The Family And Children": "Other",
    "Prostitution": "Other",
    "Rape": "Other",
    "Homicide": "Other",
    "Human Trafficking (A), Commercial Sex Acts": "Other",
    "Human Trafficking, Commercial Sex Acts": "Other",
    "Human Trafficking (B), Involuntary Servitude": "Other"
}

df["Incident Category"] = df["Incident Category"].replace(category_mapping)

valid_labels = {
    "Fight",
    "Theft",
    "Fire",
    "Medical Emergency",
    "Unauthorized Access",
    "Vandalism",
    "Fraud",
    "Suspicious Activity",
    "Weapons",
    "Other"
}

df["Incident Category"] = df["Incident Category"].apply(
    lambda x: x if x in valid_labels else "Other"
)

# ==========================================================
# MAP LOCATIONS
# ==========================================================

location_mapping = {

    "Northern": "Library",
    "Southern": "Main Gate",
    "Central": "Administration Block",
    "Mission": "Hostel A",
    "Bayview": "Parking Area",
    "Richmond": "Computer Lab",
    "Taraval": "Playground",
    "Ingleside": "Engineering Block",
    "Tenderloin": "Cafeteria",

    "Park": "Playground",
    "Out of SF": "Main Gate"
}

df["Police District"] = df["Police District"].replace(location_mapping)

# Fill missing locations
df["Police District"] = df["Police District"].fillna("Unknown")

# ==========================================================
# RENAME COLUMNS
# ==========================================================

df.rename(columns={
    "Incident Category": "Label",
    "Incident Date": "Date",
    "Incident Time": "Time",
    "Police District": "Location"
}, inplace=True)

# ==========================================================
# CREATE NATURAL LANGUAGE REPORT
# ==========================================================

df["Report"] = (
    "Incident reported: "
    + df["Incident Description"].fillna("").astype(str)
    + ". Location: "
    + df["Location"].fillna("").astype(str)
    + ". Date: "
    + df["Date"].fillna("").astype(str)
    + ". Time: "
    + df["Time"].fillna("").astype(str)
)

# Remove the original incident description column
df.drop(columns=["Incident Description"], inplace=True)

# ==========================================================
# REORDER COLUMNS
# ==========================================================

df = df[
    [
        "Report",
        "Label",
        "Date",
        "Time",
        "Location"
    ]
]


# ==========================================================
# REMOVE EMPTY RECORDS
# ==========================================================

df = df.dropna(subset=["Report", "Label"])

# Remove duplicate records
df = df.drop_duplicates()

# ==========================================================
# SAVE DATASET
# ==========================================================

df.to_csv(output_file, index=False)

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print("\nDataset Created Successfully!\n")

print(df.head(10))

print("\n")

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Total Records : {len(df)}")
print(f"Total Classes : {df['Label'].nunique()}")

print("\nClasses:")

print(df["Label"].value_counts())

print("\nSaved To:")

print(output_file)

print("\nDone Successfully!")