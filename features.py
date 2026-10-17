import math

import numpy as np

# ----------------------------
# Feature engineering shared by training (train_model.py) and the app
# ----------------------------
# Banks judge affordability (how much of your income goes to EMIs) rather
# than the raw loan amount, and treat credit score in bands rather than as a
# straight line. The features below follow that.

# Interest rate assumed when estimating the EMI of the requested loan
ANNUAL_RATE = 0.105

# Credit-score bands. In the training data approvals jump at 650, which
# also matches the minimum score Indian banks commonly quote. Approval
# rates are flat above 700 in the data, so there is no 750 band (it came
# out slightly negative, which would make a higher score look worse).
CREDIT_BANDS = [600, 650, 700]

# Range of total monthly income seen in the training data. Values outside
# are clipped so the model never extrapolates.
INCOME_RANGE = (15000, 200000)

# The EMI burden (FOIR) is deliberately NOT a model feature: every loan in
# the training data costs under 20% of income (about 3% on average) and
# approval rates don't change with it, so the model would only learn to
# punish normal burdens. Affordability is checked by the bank policy rules
# instead (policy.py).

EMPLOYMENT = {0: "Salaried", 1: "Self-employed", 2: "Unemployed"}
AREAS = {0: "Rural", 1: "Semiurban", 2: "Urban"}

FEATURES = (
    [f"cs_ge_{b}" for b in CREDIT_BANDS]
    + [
        "log_total_income",
        "dependents",
        "graduate",
        "self_employed",
        "unemployed",
        "property_area",
    ]
)


def emi(principal, months, annual_rate=ANNUAL_RATE):
    """Monthly instalment for a loan (works on numbers and numpy arrays)."""
    r = annual_rate / 12
    growth = (1 + r) ** months
    return principal * r * growth / (growth - 1)


def total_income(values):
    return values["applicant_income"] + values["coapplicant_income"]


def foir(values):
    """
    Fixed Obligations to Income Ratio: all monthly EMIs (existing plus the
    new loan) divided by total monthly income.
    """
    obligations = values.get("existing_emi", 0) + emi(
        values["loan_amount"], values["loan_term"])
    income = total_income(values)
    return obligations / income if income > 0 else math.inf


def build_features(values):
    """
    Turn one application (dict of form values) or a dict of numpy arrays
    (the training data) into the model's feature matrix.
    """
    cs = np.asarray(values["credit_score"], dtype=float)
    income = np.asarray(values["applicant_income"], dtype=float) + \
        np.asarray(values["coapplicant_income"], dtype=float)
    employment = np.asarray(values["employment"])

    columns = [(cs >= b).astype(float) for b in CREDIT_BANDS]
    columns += [
        np.log(np.clip(income, *INCOME_RANGE)),
        np.asarray(values["dependents"], dtype=float),
        np.asarray(values["education"], dtype=float),
        (employment == 1).astype(float),
        (employment == 2).astype(float),
        np.asarray(values["property_area"], dtype=float),
    ]
    return np.column_stack([np.atleast_1d(c) for c in columns])
