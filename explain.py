import math

import numpy as np

# ----------------------------
# Explainability for the loan model
# ----------------------------
# The model is StandardScaler -> LogisticRegression, so its decision score
# (log-odds) is a sum of one term per feature:
#
#     score = intercept + sum(coef_i * (x_i - mean_i) / std_i)
#
# Each term is that feature's exact contribution compared with the average
# applicant in the training data (for a linear model this equals its SHAP
# value). Positive terms push towards approval, negative terms towards
# rejection.

FEATURES = [
    "gender",
    "married",
    "dependents",
    "education",
    "applicant_income",
    "coapplicant_income",
    "loan_amount",
    "loan_term",
    "property_area",
    "credit_score",
    "self_employed",
]

LABELS = {
    "gender": "Gender",
    "married": "Marital status",
    "dependents": "Dependents",
    "education": "Education",
    "applicant_income": "Applicant income",
    "coapplicant_income": "Co-applicant income",
    "loan_amount": "Loan amount",
    "loan_term": "Loan term",
    "property_area": "Property area",
    "credit_score": "Credit score",
    "self_employed": "Employment type",
}

# Gender and marital status should not drive a lending decision; they are
# still shown in the breakdown so the explanation stays honest about what
# the model used, but they are never given as a reason.
PROTECTED = {"gender", "married"}

AREAS = {0: "Rural", 1: "Semiurban", 2: "Urban"}

# Slider limits in the form, used to keep suggestions within reach
LIMITS = {
    "credit_score": (300, 900),
    "loan_amount": (50000, 5000000),
}


def inr(value):
    """Format a number in Indian style, e.g. 1500000 -> ₹15,00,000."""
    value = int(round(value))
    s = str(abs(value))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return ("-" if value < 0 else "") + "₹" + s


def display_value(name, value):
    if name == "gender":
        return "Male" if value == 1 else "Female"
    if name == "married":
        return "Married" if value == 1 else "Unmarried"
    if name == "education":
        return "Graduate" if value == 1 else "Not graduate"
    if name == "self_employed":
        return "Self-employed" if value == 1 else "Salaried"
    if name == "property_area":
        return AREAS.get(value, str(value))
    if name in ("applicant_income", "coapplicant_income"):
        return inr(value) + "/month"
    if name == "loan_amount":
        return inr(value)
    if name == "loan_term":
        return f"{value} months"
    return str(value)


def describe(name, value, mean, helps):
    """Plain-language sentence for one factor."""
    if name == "credit_score":
        side = "above" if value > mean else "below"
        return (f"Your credit score of {value} is {side} the typical "
                f"applicant's score of about {mean:.0f}. Credit score is the "
                f"factor this model weighs most heavily.")
    if name == "applicant_income":
        side = "higher" if value > mean else "lower"
        return (f"Your monthly income of {inr(value)} is {side} than the "
                f"typical {inr(mean)}, so repayment capacity looks "
                f"{'stronger' if helps else 'weaker'}.")
    if name == "coapplicant_income":
        if value == 0:
            return ("There is no co-applicant income to support repayment "
                    f"(typical applicants have about {inr(mean)}/month).")
        side = "higher" if value > mean else "lower"
        return (f"Co-applicant income of {inr(value)}/month is {side} than "
                f"the typical {inr(mean)}.")
    if name == "loan_amount":
        side = "larger" if value > mean else "smaller"
        return (f"The requested loan of {inr(value)} is {side} than the "
                f"typical {inr(mean)}, which "
                f"{'raises' if not helps else 'lowers'} the lending risk.")
    if name == "loan_term":
        side = "longer" if value > mean else "shorter"
        return (f"A loan term of {value} months is {side} than the typical "
                f"{mean:.0f} months, which "
                f"{'eases' if helps else 'increases'} the monthly burden.")
    if name == "dependents":
        if helps:
            who = "no dependents" if value == 0 else f"{value} dependent(s)"
            return (f"Having {who} means fewer people relying on your "
                    f"income than the typical applicant.")
        return (f"Having {value} dependents means more people relying on "
                f"your income than the typical applicant.")
    if name == "education":
        return ("Being a graduate counts in your favour." if value == 1
                else "Not being a graduate counts against the application.")
    if name == "self_employed":
        return ("A salaried job counts as a more stable income source."
                if value == 0 else
                "Self-employed income is treated as less stable than a "
                "salaried job.")
    if name == "property_area":
        area = AREAS.get(value, str(value))
        return (f"The property is in a {area.lower()} area, which "
                f"{'helps' if helps else 'counts against'} the application.")
    if name == "married":
        return "Marital status had a small effect on the model's score."
    if name == "gender":
        return "Gender had a very small effect on the model's score."
    return f"{LABELS[name]} {'helped' if helps else 'hurt'} the application."


def impact_label(contribution, biggest):
    share = abs(contribution) / biggest if biggest else 0
    if share >= 0.5:
        return "High"
    if share >= 0.15:
        return "Medium"
    return "Low"


def explain(model, values, threshold):
    """
    Build an explanation for one application.

    values: dict of feature name -> int, as read from the form.
    Returns a dict with the probability, every factor's contribution,
    the main reasons for the decision, and what could change it.
    """
    scaler = model.named_steps["scaler"]
    lr = model.named_steps["lr"]

    x = np.array([values[f] for f in FEATURES], dtype=float)
    mean = scaler.mean_
    std = scaler.scale_
    coef = lr.coef_[0]

    contributions = coef * (x - mean) / std
    score = lr.intercept_[0] + contributions.sum()
    probability = 1 / (1 + math.exp(-score))
    approved = probability >= threshold

    biggest = float(np.abs(contributions).max())
    factors = []
    for i, name in enumerate(FEATURES):
        c = float(contributions[i])
        factors.append({
            "name": name,
            "label": LABELS[name],
            "value": display_value(name, values[name]),
            "contribution": round(c, 3),
            "helps": c >= 0,
            "impact": impact_label(c, biggest),
            "width": round(abs(c) / biggest * 100, 1) if biggest else 0,
            "text": describe(name, values[name], mean[i], c >= 0),
            "protected": name in PROTECTED,
        })
    factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)

    # Reasons: the strongest factors that pushed in the direction of the
    # decision. Ignore near-zero effects so we never pad the list.
    reasons = [
        f for f in factors
        if f["helps"] == approved
        and not f["protected"]
        and abs(f["contribution"]) >= 0.05
    ][:3]

    # Counter-points: strongest factors that pushed the other way
    against = [
        f for f in factors
        if f["helps"] != approved
        and not f["protected"]
        and abs(f["contribution"]) >= 0.05
    ][:2]

    suggestions = []
    if not approved:
        suggestions = what_would_change(values, score, threshold,
                                        coef, mean, std)

    return {
        "probability": round(probability * 100, 1),
        "threshold": round(threshold * 100),
        "factors": factors,
        "reasons": reasons,
        "against": against,
        "suggestions": suggestions,
    }


def what_would_change(values, score, threshold, coef, mean, std):
    """
    For a rejected application, find how far a single input would need to
    move (with everything else unchanged) to reach the approval threshold.
    Exact, because the score is linear in each input.
    """
    target = math.log(threshold / (1 - threshold))
    gap = target - score
    suggestions = []

    i = FEATURES.index("credit_score")
    needed = values["credit_score"] + gap / coef[i] * std[i]
    low, high = LIMITS["credit_score"]
    if needed <= high:
        needed = max(low, math.ceil(needed))
        suggestions.append(
            f"Raising your credit score from {values['credit_score']} to "
            f"about {needed} would likely get this loan approved, with "
            f"everything else unchanged."
        )
    else:
        suggestions.append(
            "Even the maximum credit score would not be enough on its own "
            "for this loan; other details need to change as well."
        )

    i = FEATURES.index("loan_amount")
    needed = values["loan_amount"] + gap / coef[i] * std[i]
    low, high = LIMITS["loan_amount"]
    if low <= needed < values["loan_amount"]:
        needed = math.floor(needed / 10000) * 10000
        if needed >= low:
            suggestions.append(
                f"Alternatively, reducing the loan amount from "
                f"{inr(values['loan_amount'])} to about {inr(needed)} "
                f"would likely get it approved."
            )

    return suggestions
