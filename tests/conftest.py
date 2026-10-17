import os
import pickle
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# A strong application that passes every policy check and the model
APPROVED = {
    "age": 30,
    "dependents": 0,
    "education": 1,
    "employment": 0,
    "property_area": 2,
    "credit_score": 750,
    "applicant_income": 60000,
    "coapplicant_income": 0,
    "existing_emi": 0,
    "loan_amount": 500000,
    "loan_term": 60,
}

# Rejected applications, each failing for a different reason
REJECTED = {
    "low credit score": dict(APPROVED, credit_score=600),
    "unaffordable loan": dict(APPROVED, loan_amount=3000000),
    "unemployed alone": dict(APPROVED, employment=2, applicant_income=20000),
    "too old at maturity": dict(APPROVED, age=58, loan_term=180),
    "many problems": dict(APPROVED, credit_score=500, loan_amount=5000000,
                          applicant_income=10000),
}


@pytest.fixture(scope="session")
def model():
    with open(os.path.join(ROOT, "loan_model.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.fixture
def approved():
    return dict(APPROVED)
