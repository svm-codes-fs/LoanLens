import pytest

from app import app
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
