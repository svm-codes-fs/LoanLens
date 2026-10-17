import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, build_features

THRESHOLD = 0.50

# Load data
data = pd.read_excel("train_with_credit_score.xlsx")
data.columns = data.columns.str.strip().str.replace("-", "_")

# ----------------------------
# Clean
# ----------------------------
# Not used:
#   Loan_ID         identifier
#   Credit_History  leaks the label (0 -> 99.9% rejected)
#   Age             handled by the bank policy rules instead (policy.py)
#   Gender, Married lending decisions should not depend on them
#
# Missing incomes / loan amounts are filled with the median, not 0, so they
# don't look like applicants with no income.
for col in ["Applicant_Income", "Loan_Amount"]:
    data[col] = data[col].fillna(data[col].median())

values = {
    "credit_score": data["Credit_Score"].to_numpy(),
    "applicant_income": data["Applicant_Income"].to_numpy(),
    "coapplicant_income": data["Coapplicant_Income"].fillna(0).to_numpy(),
    "loan_amount": data["Loan_Amount"].to_numpy(),
    "loan_term": data["Loan_Term"].to_numpy(),
    "dependents": data["Dependents"].fillna(0).to_numpy(),
    # Missing education is treated as not graduate (the cautious choice)
    "education": (data["Education"] == "Graduate").astype(int).to_numpy(),
    # 0 salaried, 1 self-employed, 2 unemployed; missing -> salaried (mode)
    "employment": data["Employment_Status"].map({
        "Salaried": 0, "Self-Employed": 1, "Unemployed": 2
    }).fillna(0).astype(int).to_numpy(),
    "property_area": data["Property_Area"].map({
        "Rural": 0, "Semiurban": 1, "Urban": 2
    }).to_numpy(),
}

X = build_features(values)
y = (data["Loan_Status"] == "Approved").astype(int).to_numpy()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000)),
])

# ----------------------------
# Evaluate
# ----------------------------
cv = StratifiedKFold(5, shuffle=True, random_state=42)
auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"5-fold CV  ROC-AUC {auc.mean():.3f}  accuracy {acc.mean():.3f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
print(f"Hold-out (30%) at threshold {THRESHOLD}:")
print(f"  ROC-AUC   {roc_auc_score(y_test, proba):.3f}")
print(f"  Accuracy  {accuracy_score(y_test, pred):.3f}")
print(f"  Precision {precision_score(y_test, pred):.3f}")
print(f"  Recall    {recall_score(y_test, pred):.3f}")
print(f"  F1        {f1_score(y_test, pred):.3f}")

# ----------------------------
# Fit on all data and save
# ----------------------------
model.fit(X, y)
coef = model.named_steps["lr"].coef_[0]
for name, c in sorted(zip(FEATURES, coef), key=lambda t: -abs(t[1])):
    print(f"  {name:18s} {c:+.2f}")

with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model retrained and saved")
