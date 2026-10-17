import pandas as pd

from pathlib import Path

# Paths are relative to this file, so the script runs from any folder
HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "train_with_credit_score.xlsx"

# -----------------------------
# Day 2: Data Understanding
# -----------------------------

# Load dataset (UPDATED)
data = pd.read_excel(DATASET)

print("✅ Dataset loaded successfully\n")

# 1. Shape of dataset
print("🔹 Dataset Shape (rows, columns):")
print(data.shape, "\n")

# 2. Column names
print("🔹 Column Names:")
print(list(data.columns), "\n")

# 3. Data types
print("🔹 Data Types:")
print(data.dtypes, "\n")

# 4. First 5 rows
print("🔹 Sample Data:")
print(data.head(), "\n")

# 5. Missing values check
print("🔹 Missing Values per Column:")
print(data.isnull().sum(), "\n")

# 6. Basic statistics (numeric columns)
print("🔹 Statistical Summary:")
print(data.describe())
