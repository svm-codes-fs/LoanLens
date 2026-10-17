from flask import Flask, jsonify, render_template, request
import os
import pickle

from explain import explain

app = Flask(__name__)

# ----------------------------
# Load trained model
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "loan_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FIELDS = [
    "age", "dependents", "education", "employment", "property_area",
    "credit_score", "applicant_income", "coapplicant_income",
    "existing_emi", "loan_amount", "loan_term",
]


def read_values(source):
    """Read every field from a form or JSON dict as an int."""
    return {name: int(source.get(name)) for name in FIELDS}


# ----------------------------
# Home route
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        show_result=False
    )


# ----------------------------
# Prediction route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # ----------------------------
        # Read form values safely
        # ----------------------------
        values = read_values(request.form)

        # ----------------------------
        # Decide and explain (policy rules + model, see explain.py)
        # ----------------------------
        explanation = explain(model, values)
        status = "approved" if explanation["approved"] else "rejected"

        # ----------------------------
        # Render result page (STEP 3)
        # ----------------------------
        return render_template(
            "index.html",
            show_result=True,
            status=status,
            explanation=explanation
        )

    except Exception as e:
        # Fallback — go back safely
        print("Prediction error:", e)
        return render_template(
            "index.html",
            show_result=False
        )


# ----------------------------
# JSON API
# ----------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON object with the application fields."), 400

    missing = [name for name in FIELDS if data.get(name) is None]
    if missing:
        return jsonify(error="Missing fields.", missing=missing), 400
    try:
        values = read_values(data)
    except (TypeError, ValueError):
        return jsonify(error="Every field must be an integer."), 400

    explanation = explain(model, values)
    return jsonify(
        decision="approved" if explanation["approved"] else "rejected",
        probability=explanation["probability"],
        threshold=explanation["threshold"],
        checks=explanation["checks"],
        # Every reason in the same shape, without the chart-only fields
        reasons=[{k: r[k] for k in ("label", "value", "text")}
                 for r in explanation["reasons"]],
        suggestions=explanation["suggestions"],
    )


if __name__ == "__main__":
    app.run(debug=True)