from flask import Flask, render_template, request
import pickle
import numpy as np

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
        gender = int(request.form.get("gender"))
        married = int(request.form.get("married"))
        dependents = int(request.form.get("dependents"))
        education = int(request.form.get("education"))
        self_employed = int(request.form.get("self_employed"))
        property_area = int(request.form.get("property_area"))

        credit_score = int(request.form.get("credit_score"))
        applicant_income = int(request.form.get("applicant_income"))
        coapplicant_income = int(request.form.get("coapplicant_income"))
        loan_amount = int(request.form.get("loan_amount"))
        loan_term = int(request.form.get("loan_term"))

        # ----------------------------
        # Feature vector (ORDER MATTERS)
        # ----------------------------
        features = np.array([[
            gender,
            married,
            dependents,
            education,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            property_area,
            credit_score,
            self_employed
        ]])

        # ----------------------------
        # PROBABILITY-BASED DECISION
        # ----------------------------
        approval_probability = model.predict_proba(features)[0][1]

        # Custom threshold (REALISTIC)
        if approval_probability >= 0.40:
            status = "approved"
        else:
            status = "rejected"

        # ----------------------------
        # Render result page (STEP 3)
        # ----------------------------
        return render_template(
            "index.html",
            show_result=True,
            status=status
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