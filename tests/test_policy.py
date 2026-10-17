import math

import pytest

from features import emi
from policy import (MAX_AGE, MAX_AGE_AT_MATURITY, MIN_AGE, MIN_CREDIT_SCORE,
                    MIN_MONTHLY_INCOME, check_rules, inr)


def passed(values, name):
    """Whether the rule called `name` passes for these values."""
    return {c["name"]: c["passed"] for c in check_rules(values)}[name]


def test_every_rule_passes_for_a_strong_application(approved):
    checks = check_rules(approved)
    assert len(checks) == 5
    assert all(c["passed"] for c in checks)
    assert all(c["text"] for c in checks)


@pytest.mark.parametrize("score, ok", [
    (MIN_CREDIT_SCORE, True),
    (MIN_CREDIT_SCORE - 1, False),
])
def test_minimum_credit_score(approved, score, ok):
    approved["credit_score"] = score
    assert passed(approved, "Minimum credit score") is ok


@pytest.mark.parametrize("applicant, coapplicant, ok", [
    (MIN_MONTHLY_INCOME, 0, True),
    (MIN_MONTHLY_INCOME - 1, 0, False),
    # Co-applicant income counts towards the household total
    (MIN_MONTHLY_INCOME - 5000, 5000, True),
    (MIN_MONTHLY_INCOME - 5000, 4999, False),
])
def test_minimum_income(approved, applicant, coapplicant, ok):
    approved.update(applicant_income=applicant,
                    coapplicant_income=coapplicant)
    assert passed(approved, "Minimum income") is ok


@pytest.mark.parametrize("employment, coapplicant, ok", [
    (0, 0, True),    # salaried
    (1, 0, True),    # self-employed
    (2, 0, False),   # unemployed with no co-applicant income
    (2, 1, True),    # unemployed with a co-applicant who earns
])
def test_source_of_income(approved, employment, coapplicant, ok):
    approved.update(employment=employment, coapplicant_income=coapplicant)
    assert passed(approved, "Source of income") is ok


def test_affordability_limit(approved):
    # Existing EMIs that bring the total just under / just over 50%
    room = 0.50 * approved["applicant_income"] - emi(
        approved["loan_amount"], approved["loan_term"])
    approved["existing_emi"] = math.floor(room)
    assert passed(approved, "Affordability (EMI to income)")
    approved["existing_emi"] = math.ceil(room) + 1
    assert not passed(approved, "Affordability (EMI to income)")


def test_affordability_fails_without_income(approved):
    approved.update(applicant_income=0, coapplicant_income=0)
    assert not passed(approved, "Affordability (EMI to income)")


@pytest.mark.parametrize("age, ok", [
    (MIN_AGE, True),
    (MIN_AGE - 1, False),
    (MAX_AGE, True),
    (MAX_AGE + 1, False),
])
def test_age_limits(approved, age, ok):
    approved.update(age=age, loan_term=12)
    assert passed(approved, "Age") is ok


@pytest.mark.parametrize("term, ok", [
    ((MAX_AGE_AT_MATURITY - 40) * 12, True),        # ends exactly at 65
    ((MAX_AGE_AT_MATURITY - 40) * 12 + 12, False),  # ends at 66
])
def test_age_at_maturity(approved, term, ok):
    approved.update(age=40, loan_term=term)
    assert passed(approved, "Age") is ok


@pytest.mark.parametrize("value, text", [
    (0, "₹0"),
    (999, "₹999"),
    (15000, "₹15,000"),
    (1500000, "₹15,00,000"),
    (123456789, "₹12,34,56,789"),
    (-25000, "-₹25,000"),
])
def test_inr_uses_indian_grouping(value, text):
    assert inr(value) == text
