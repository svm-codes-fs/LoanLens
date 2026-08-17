import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# -----------------------------
# Day 6: Model Evaluation (FIXED)
# -----------------------------

# Load encoded dataset
data = pd.read_csv("loan_encoded.csv")
print("Encoded dataset loaded\n")

# -----------------------------
# Match TRAINING preprocessing
# -----------------------------

# Drop columns NOT used during training
columns_to_drop = []
for col in ["Loan_ID", "Age", "Credit_History"]:
    if col in data.columns:
        columns_to_drop.append(col)

data = data.drop(columns=columns_to_drop)

# Safety: fill NaNs
data = data.fillna(0)

# -----------------------------
# Split features & target
# -----------------------------
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

print("Evaluation features:")
print(list(X.columns))

# -----------------------------
# Train-test split (same seed)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# -----------------------------
# Load trained model
# -----------------------------
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

print("\nTrained model loaded successfully")

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nMODEL ACCURACY: {accuracy:.4f}")