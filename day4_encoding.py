import pandas as pd

# -----------------------------
# Day 4: Encode Categorical Data
# -----------------------------

# Load cleaned dataset from Day 3
data = pd.read_excel("loan_cleaned.xlsx")

print("Dataset loaded for encoding\n")

# Clean column names (safety step)
data.columns = (
    data.columns
        .str.strip()
        .str.replace("-", "_")
        .str.replace("\r", "")
        .str.replace("\n", "")
)

# -----------------------------
# Drop ID column
# -----------------------------
if "Loan_ID" in data.columns:
    data = data.drop("Loan_ID", axis=1)

# -----------------------------
# Encode categorical columns
# -----------------------------

# Gender
data["Gender"] = data["Gender"].map({
    "Male": 1,
    "Female": 0
})

# Married
data["Married"] = data["Married"].map({
    "Yes": 1,
    "No": 0
})

# Education
data["Education"] = data["Education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

# Employment_Status → Self_Employed
data["Self_Employed"] = data["Employment_Status"].map({
    "Self-Employed": 1,
    "Self Employed": 1,
    "Salaried": 0,
    "Employed": 0
})

# Property Area
data["Property_Area"] = data["Property_Area"].map({
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
})

# Loan Status (TARGET)
data["Loan_Status"] = data["Loan_Status"].map({
    "Approved": 1,
    "Rejected": 0
})

# -----------------------------
# Drop original categorical column
# -----------------------------
data = data.drop("Employment_Status", axis=1)

# -----------------------------
# Dependents → integer
# -----------------------------
data["Dependents"] = data["Dependents"].astype(int)

# -----------------------------
# Final check
# -----------------------------
print("Encoded dataset preview:")
print(data.head())

print("\nData types after encoding:")
print(data.dtypes)

# -----------------------------
# Save encoded dataset
# -----------------------------
data.to_csv("loan_encoded.csv", index=False)

print("\n✅ Data encoded successfully and saved as loan_encoded.csv")
