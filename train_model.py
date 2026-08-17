import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_excel("train_with_credit_score.xlsx")

# Clean column names
data.columns = data.columns.str.strip().str.replace("-", "_")

# Drop leakage / noisy columns
drop_cols = ["Loan_ID", "Age", "Credit_History"]
data = data.drop(columns=[c for c in drop_cols if c in data.columns])

# Encode categorical
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["Married"] = data["Married"].map({"Yes": 1, "No": 0})
data["Education"] = data["Education"].map({"Graduate": 1, "Not Graduate": 0})
data["Property_Area"] = data["Property_Area"].map({
    "Rural": 0, "Semiurban": 1, "Urban": 2
})
data["Self_Employed"] = data["Employment_Status"].map({
    "Self-Employed": 1, "Self Employed": 1,
    "Salaried": 0, "Employed": 0
})
data["Loan_Status"] = data["Loan_Status"].map({
    "Approved": 1, "Rejected": 0
})

data = data.drop("Employment_Status", axis=1)
data = data.fillna(0)

# Features / target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y   # 🔑 KEY FIX
)

# Pipeline (SCALING + WEAKER MODEL)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        C=0.3,                # 🔻 weaker model → lower accuracy
        class_weight="balanced",
        max_iter=1000
    ))
])

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc:.4f}")

# Save
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model retrained and saved")