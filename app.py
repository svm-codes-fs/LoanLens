from flask import Flask, render_template, request
import pickle

from explain import explain

app = Flask(__name__)

# ----------------------------
# Load trained model
# ----------------------------
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)


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
        fields = [
            "age", "dependents", "education", "employment", "property_area",
            "credit_score", "applicant_income", "coapplicant_income",
            "existing_emi", "loan_amount", "loan_term",
        ]
        values = {name: int(request.form.get(name)) for name in fields}

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


if __name__ == "__main__":
    app.run(debug=True)