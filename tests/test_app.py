import pytest

from app import FIELDS, app
from conftest import APPROVED, REJECTED


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def form(values):
    return {name: str(value) for name, value in values.items()}


def test_home_page_renders_the_form(client):
    response = client.get("/")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "LoanLens" in page
    assert 'action="/predict"' in page
    assert "educational project" in page
    assert "Loan Approved" not in page and "Loan Rejected" not in page


def test_predict_shows_approval(client):
    response = client.post("/predict", data=form(APPROVED))
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Loan Approved" in page
    assert "Why it was approved" in page


def test_predict_shows_rejection_with_reasons(client):
    response = client.post("/predict", data=form(REJECTED["low credit score"]))
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Loan Rejected" in page
    assert "Why it was rejected" in page
    assert "Minimum credit score" in page
    assert "What could change the decision" in page


def test_predict_with_missing_fields_returns_to_the_form(client):
    response = client.post("/predict", data={"age": "30"})
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Loan Approved" not in page and "Loan Rejected" not in page


API_KEYS = {"decision", "probability", "threshold", "checks", "reasons",
            "suggestions"}


def test_api_approves_a_strong_application(client):
    response = client.post("/api/predict", json=APPROVED)
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == API_KEYS
    assert body["decision"] == "approved"
    assert body["probability"] >= body["threshold"]
    assert all(c["passed"] for c in body["checks"])
    assert body["suggestions"] == []


def test_api_explains_a_rejection(client):
    response = client.post("/api/predict", json=REJECTED["low credit score"])
    assert response.status_code == 200
    body = response.get_json()
    assert body["decision"] == "rejected"
    assert body["reasons"][0] == {
        "label": "Minimum credit score",
        "value": "Not met",
        "text": next(c["text"] for c in body["checks"]
                     if c["name"] == "Minimum credit score"),
    }
    assert any("credit score" in s for s in body["suggestions"])


def test_api_matches_the_explain_function(client, model):
    from explain import explain

    values = REJECTED["unaffordable loan"]
    expected = explain(model, values)
    body = client.post("/api/predict", json=values).get_json()
    for key in ["probability", "threshold", "checks", "suggestions"]:
        assert body[key] == expected[key]
    assert body["reasons"] == [
        {"label": r["label"], "value": r["value"], "text": r["text"]}
        for r in expected["reasons"]
    ]


def test_api_accepts_numeric_strings(client):
    values = {k: str(v) for k, v in APPROVED.items()}
    response = client.post("/api/predict", json=values)
    assert response.status_code == 200
    assert response.get_json()["decision"] == "approved"


def test_api_lists_missing_fields(client):
    response = client.post("/api/predict", json={"age": 30})
    assert response.status_code == 400
    assert response.get_json()["missing"] == [f for f in FIELDS if f != "age"]


@pytest.mark.parametrize("payload", [
    None,                                  # no body at all
    [1, 2, 3],                             # not an object
    dict(APPROVED, credit_score="high"),   # not a number
])
def test_api_rejects_bad_input(client, payload):
    if payload is None:
        response = client.post("/api/predict", data="not json",
                               content_type="application/json")
    else:
        response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "error" in response.get_json()
