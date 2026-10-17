import json
import re

import pytest

from conftest import APPROVED, REJECTED
from explain import THRESHOLD, decide, explain, what_would_change
from features import build_features

ALL_CASES = dict(REJECTED, approved=APPROVED)


@pytest.mark.parametrize("values", ALL_CASES.values(), ids=ALL_CASES.keys())
def test_decide_matches_model_predict_proba(model, values):
    _, p, _ = decide(model, values)
    expected = model.predict_proba(build_features(values))[0, 1]
    assert p == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("values", ALL_CASES.values(), ids=ALL_CASES.keys())
def test_decide_needs_every_check_and_the_threshold(model, values):
    approved, p, checks = decide(model, values)
    assert approved == (all(c["passed"] for c in checks) and p >= THRESHOLD)


def test_strong_application_is_approved(model):
    result = explain(model, APPROVED)
    assert result["approved"]
    assert result["suggestions"] == []
    assert result["reasons"][0]["label"] == "Affordability"


@pytest.mark.parametrize("values", REJECTED.values(), ids=REJECTED.keys())
def test_rejected_applications_are_rejected(model, values):
    assert not explain(model, values)["approved"]


# Each suggestion names one field and the new value it needs
SUGGESTION_PATTERNS = [
    (r"credit score from \d+ to (\d+)", "credit_score"),
    (r"loan amount from ₹[\d,]+ to ₹([\d,]+)", "loan_amount"),
    (r"loan term from \d+ to (\d+) months", "loan_term"),
    (r"co-applicant income of ₹([\d,]+)/month", "coapplicant_income"),
]


def parse_suggestion(text):
    """Return (field, new value) for a suggestion, or None."""
    for pattern, field in SUGGESTION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return field, int(match.group(1).replace(",", ""))
    return None


@pytest.mark.parametrize("values", REJECTED.values(), ids=REJECTED.keys())
def test_every_suggestion_gets_the_loan_approved(model, values):
    suggestions = what_would_change(model, values)
    assert suggestions
    for text in suggestions:
        change = parse_suggestion(text)
        if change is None:
            assert text.startswith("No single change")
            continue
        field, new_value = change
        assert decide(model, dict(values, **{field: new_value}))[0], text


@pytest.mark.parametrize("name", [
    "low credit score", "unaffordable loan", "unemployed alone",
    "too old at maturity",
])
def test_suggestions_are_found_for_fixable_rejections(model, name):
    suggestions = what_would_change(model, REJECTED[name])
    assert any(parse_suggestion(s) for s in suggestions)


def test_suggested_credit_score_is_the_smallest_that_works(model):
    values = REJECTED["low credit score"]
    field, needed = parse_suggestion(what_would_change(model, values)[0])
    assert field == "credit_score"
    assert not decide(model, dict(values, credit_score=needed - 1))[0]


@pytest.mark.parametrize("values", REJECTED.values(), ids=REJECTED.keys())
def test_reasons_lead_with_failed_checks(model, values):
    result = explain(model, values)
    failed = [c["name"] for c in result["checks"] if not c["passed"]]
    assert failed
    leading = result["reasons"][:len(failed)]
    assert [r["label"] for r in leading] == failed
    assert all(r["value"] == "Not met" for r in leading)


def test_failed_check_is_not_repeated_as_a_model_reason(model):
    result = explain(model, REJECTED["low credit score"])
    labels = [r["label"] for r in result["reasons"]]
    assert "Minimum credit score" in labels
    assert "Credit score" not in labels


@pytest.mark.parametrize("values", ALL_CASES.values(), ids=ALL_CASES.keys())
def test_explanation_is_json_serialisable(model, values):
    json.dumps(explain(model, values))
