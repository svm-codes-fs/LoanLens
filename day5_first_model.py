import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# -----------------------------
# Day 5: Model Training
# -----------------------------

# Load encoded dataset
data = pd.read_csv("loan_encoded.csv")

print("Encoded dataset loaded\n")

# -----------------------------
# FINAL NaN CHECK (CRITICAL)
# -----------------------------
print("Missing values BEFORE fixing:")
print(data.isnull().sum(), "\n")

# Fill any remaining NaN values
data = data.fillna(0)

print("Missing values AFTER fixing:")
print(data.isnull().sum(), "\n")

# -----------------------------
# Features & Target
# -----------------------------
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

print("Training features:")
print(list(X.columns), "\n")

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# -----------------------------
# Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# -----------------------------
# Predictions & Accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("DAY 5 RESULT")
print("Model Accuracy:", round(accuracy, 4))
print("Predictions (1 = Approved, 0 = Rejected):")
print(y_pred)

# -----------------------------
# Save trained model
# -----------------------------
with open("loan_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n✅ Trained model saved as loan_model.pkl")
