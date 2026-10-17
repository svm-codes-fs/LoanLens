import pandas as pd

from pathlib import Path

# Paths are relative to this file, so the script runs from any folder
HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "train_with_credit_score.xlsx"

# -----------------------------
# Day 3: Handle Missing Values
# -----------------------------

# Load dataset (UPDATED)
data = pd.read_excel(DATASET)

# Clean column names (important)
data.columns = (
    data.columns
        .str.strip()
        .str.replace("-", "_")
        .str.replace("\r", "")
        .str.replace("\n", "")
)

print("Dataset loaded\n")

# -----------------------------
# Numerical columns
# -----------------------------
numerical_cols = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Loan_Amount",
    "Loan_Term",
    "Credit_Score"
]

for col in numerical_cols:
    if col in data.columns:
        data[col].fillna(data[col].median(), inplace=True)

# -----------------------------
# Categorical columns
# -----------------------------
categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Employment_Status",
    "Property_Area",
    "Loan_Status"
]

for col in categorical_cols:
    if col in data.columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

# -----------------------------
# Missing value check
# -----------------------------
print("Missing values AFTER fixing:")
print(data.isnull().sum())

# -----------------------------
# Save cleaned dataset
# -----------------------------
data.to_excel(HERE / "loan_cleaned.xlsx", index=False)

print("\n✅ Missing values handled. Cleaned dataset saved as loan_cleaned.xlsx")
