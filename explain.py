import math

import numpy as np

from features import (AREAS, CREDIT_BANDS, EMPLOYMENT, FEATURES,
                      build_features, foir, total_income)
from policy import MAX_FOIR, check_rules, inr

# ----------------------------
# Decision + explanation (XAI)
# ----------------------------
# A loan is approved only if it passes every bank policy rule (policy.py)
# AND the model's approval probability reaches the threshold.
#
# The model is StandardScaler -> LogisticRegression, so its score (log-odds)
# is a sum of one term per feature:
#
#     score = intercept + sum(coef_i * (x_i - mean_i) / std_i)
#
# Each term is that feature's exact contribution compared with the average
# applicant in the training data (for a linear model this equals its SHAP
# value). Terms are grouped into the factors an applicant recognises, e.g.
# the three credit-score bands become one "Credit score" factor.

THRESHOLD = 0.50

GROUPS = {
    "credit_score": [f"cs_ge_{b}" for b in CREDIT_BANDS],
    "income": ["log_total_income"],
    "dependents": ["dependents"],
    "education": ["graduate"],
    "employment": ["self_employed", "unemployed"],
    "property_area": ["property_area"],
}

LABELS = {
    "credit_score": "Credit score",
    "income": "Household income",
    "dependents": "Dependents",
    "education": "Education",
    "employment": "Employment",
    "property_area": "Property area",
}

# A failed rule already explains its factor, so that factor is not repeated
RULE_FACTOR = {
    "Minimum credit score": "credit_score",
    "Minimum income": "income",
}

# Form limits, used when searching for what would change a decision
LIMITS = {
    "credit_score": (300, 900),
    "loan_amount": (50000, 5000000),
    "loan_term": (12, 360),
}


def model_parts(model):
    scaler = model.named_steps["scaler"]
    lr = model.named_steps["lr"]
    return lr.intercept_[0], lr.coef_[0], scaler.mean_, scaler.scale_


def contributions(model, values):
    _, coef, mean, std = model_parts(model)
    x = build_features(values)[0]
    return coef * (x - mean) / std


def probability(model, values):
    intercept = model_parts(model)[0]
    score = intercept + contributions(model, values).sum()
    return 1 / (1 + math.exp(-score))


def decide(model, values):
    """Return (approved, probability, rule checks)."""
    checks = check_rules(values)
    p = probability(model, values)
    approved = all(c["passed"] for c in checks) and p >= THRESHOLD
    return approved, p, checks


def display_value(group, values):
    if group == "credit_score":
        return str(values["credit_score"])
    if group == "income":
        return inr(total_income(values)) + "/month"
    if group == "dependents":
        return str(values["dependents"])
    if group == "education":
        return "Graduate" if values["education"] == 1 else "Not graduate"
    if group == "employment":
        return EMPLOYMENT.get(values["employment"], "")
    if group == "property_area":
        return AREAS.get(values["property_area"], "")
    return ""


def describe(group, values, helps, model):
    """Plain-language sentence for one factor."""
    _, _, mean, _ = model_parts(model)
    if group == "credit_score":
        cs = values["credit_score"]
        if cs < 600:
            return (f"A credit score of {cs} is well below 650. In the "
                    f"data, almost no one below 650 was approved; this is "
                    f"the factor the model weighs most.")
        if cs < 650:
            return (f"A credit score of {cs} is just under 650, the point "
                    f"where approvals jump in the data. This is the factor "
                    f"the model weighs most.")
        if cs < 700:
            return (f"A credit score of {cs} clears the 650 mark that most "
                    f"approved applicants meet, the factor the model weighs "
                    f"most.")
        return (f"A credit score of {cs} is in the strongest band (700+), "
                f"the factor the model weighs most.")
    if group == "income":
        typical = math.exp(mean[FEATURES.index("log_total_income")])
        side = "higher" if helps else "lower"
        return (f"Household income of {inr(total_income(values))}/month is "
                f"{side} than the typical {inr(typical)}, so repayment "
                f"capacity looks {'stronger' if helps else 'weaker'}.")
    if group == "dependents":
        n = values["dependents"]
        if helps:
            who = "no dependents" if n == 0 else f"{n} dependent(s)"
            return (f"Having {who} means fewer people relying on your "
                    f"income than the typical applicant.")
        return (f"Having {n} dependents means more people relying on your "
                f"income than the typical applicant.")
    if group == "education":
        return ("Being a graduate counts in your favour."
                if values["education"] == 1 else
                "Not being a graduate counts against the application.")
    if group == "employment":
        e = values["employment"]
        if e == 0:
            return "A salaried job counts as a stable income source."
        if e == 1:
            return ("Self-employed income is treated as less stable than a "
                    "salaried job.")
        return "Being unemployed counts against the application."
    if group == "property_area":
        area = AREAS.get(values["property_area"], "").lower()
        return (f"The property is in a {area} area, which "
                f"{'helps' if helps else 'counts against'} the application.")
    return ""


def impact_label(contribution, biggest):
    share = abs(contribution) / biggest if biggest else 0
    if share >= 0.5:
        return "High"
    if share >= 0.15:
        return "Medium"
    return "Low"


def explain(model, values):
    """
    Decide on one application and explain it.

    values: dict of form values (see app.py).
    """
    approved, p, checks = decide(model, values)
    failed = [c for c in checks if not c["passed"]]

    parts = dict(zip(FEATURES, contributions(model, values)))
    grouped = {g: sum(parts[f] for f in feats) for g, feats in GROUPS.items()}
    biggest = max(abs(c) for c in grouped.values()) or 1

    factors = []
    for g, c in grouped.items():
        factors.append({
            "label": LABELS[g],
            "value": display_value(g, values),
            "contribution": round(float(c), 3),
            "helps": c >= 0,
            "impact": impact_label(c, biggest),
            "width": round(abs(c) / biggest * 100, 1),
            "text": describe(g, values, c >= 0, model),
        })
    factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)

    def strong(f):
        return abs(f["contribution"]) >= 0.05

    # Reasons: failed policy rules first, then the strongest model factors
    # that pushed in the direction of the decision.
    reasons = [{"label": c["name"], "value": "Not met", "text": c["text"]}
               for c in failed]
    covered = {LABELS[RULE_FACTOR[c["name"]]] for c in failed
               if c["name"] in RULE_FACTOR}
    model_reasons = [f for f in factors
                     if f["helps"] == approved and strong(f)
                     and f["label"] not in covered]
    if approved:
        ratio = foir(values)
        reasons.append({
            "label": "Affordability",
            "value": f"{ratio * 100:.0f}% of income",
            "text": (f"All EMIs would take {ratio * 100:.0f}% of household "
                     f"income, within the {MAX_FOIR * 100:.0f}% limit."),
        })
    reasons += model_reasons[:max(0, 3 - len(failed))]

    against = [f for f in factors if f["helps"] != approved and strong(f)][:2]

    return {
        "approved": approved,
        "probability": round(p * 100, 1),
        "threshold": round(THRESHOLD * 100),
        "checks": checks,
        "factors": factors,
        "reasons": reasons,
        "against": against,
        "suggestions": [] if approved else what_would_change(model, values),
    }


def search(model, values, field, candidates):
    """First candidate value of `field` that gets the loan approved."""
    for v in candidates:
        if decide(model, dict(values, **{field: v}))[0]:
            return v
    return None


def what_would_change(model, values):
    """
    For a rejected application, find single changes (everything else kept
    the same) that would get it approved, using the full decision: rules
    and model together.
    """
    suggestions = []

    cs = values["credit_score"]
    needed = search(model, values, "credit_score",
                    range(cs + 1, LIMITS["credit_score"][1] + 1))
    if needed:
        suggestions.append(
            f"Raising your credit score from {cs} to {needed} would get "
            f"this loan approved.")

    amount = values["loan_amount"]
    low = LIMITS["loan_amount"][0]
    step = 10000
    start = (amount - 1) // step * step
    needed = search(model, values, "loan_amount",
                    range(start, low - 1, -step))
    if needed:
        suggestions.append(
            f"Reducing the loan amount from {inr(amount)} to {inr(needed)} "
            f"would get it approved.")

    term = values["loan_term"]
    low, high = LIMITS["loan_term"]
    # Try the nearest terms first, longer and shorter alternately
    options = sorted((t for t in range(low, high + 1, 12) if t != term),
                     key=lambda t: abs(t - term))
    needed = search(model, values, "loan_term", options)
    if needed:
        suggestions.append(
            f"Changing the loan term from {term} to {needed} months would "
            f"get it approved.")

    co = values["coapplicant_income"]
    start = (co // 5000 + 1) * 5000
    needed = search(model, values, "coapplicant_income",
                    range(start, 1000001, 5000))
    if needed:
        suggestions.append(
            f"Adding co-applicant income of {inr(needed)}/month"
            f"{' (now ' + inr(co) + ')' if co else ''} would get it "
            f"approved.")

    if not suggestions:
        suggestions.append(
            "No single change to credit score, loan amount or term would be "
            "enough on its own; several details need to improve together.")
    return suggestions
