import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Day 7: Model Comparison
# -----------------------------

# Load encoded dataset (UPDATED)
data = pd.read_csv("loan_encoded.csv")

print("Encoded dataset loaded\n")

# -----------------------------
# Final NaN safety
# -----------------------------
data = data.fillna(0)

# -----------------------------
# Features & target
# -----------------------------
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

print("Features used for training:")
print(list(X.columns), "\n")

# -----------------------------
# Same split for fair comparison
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# -------------------------
# Logistic Regression
# -------------------------
lr_model = LogisticRegression(max_iter=3000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("LOGISTIC REGRESSION RESULTS")
print("Accuracy:", round(lr_accuracy, 4))
print(classification_report(y_test, lr_pred))

# -------------------------
# Random Forest
# -------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", round(rf_accuracy, 4))
print(classification_report(y_test, rf_pred))
