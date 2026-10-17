from features import emi, foir, total_income

# ----------------------------
# Bank policy rules
# ----------------------------
# Real lenders run hard eligibility checks before (or alongside) any risk
# model. The training data cannot teach these: every loan in it is small
# compared with income (EMIs under 20% of income), so a model trained on it
# has never seen an unaffordable loan. These limits are indicative values
# commonly quoted for Indian personal loans, not any one bank's policy.

MIN_CREDIT_SCORE = 650
MIN_MONTHLY_INCOME = 15000
MAX_FOIR = 0.50
MIN_AGE = 21
MAX_AGE = 60
MAX_AGE_AT_MATURITY = 65


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


def check_rules(values):
    """Return a list of {name, passed, text} for every policy rule."""
    checks = []

    cs = values["credit_score"]
    checks.append({
        "name": "Minimum credit score",
        "passed": cs >= MIN_CREDIT_SCORE,
        "text": (f"Credit score {cs} meets the minimum of {MIN_CREDIT_SCORE}."
                 if cs >= MIN_CREDIT_SCORE else
                 f"Credit score {cs} is below the minimum of "
                 f"{MIN_CREDIT_SCORE} that lenders usually require."),
    })

    income = total_income(values)
    checks.append({
        "name": "Minimum income",
        "passed": income >= MIN_MONTHLY_INCOME,
        "text": (f"Household income of {inr(income)}/month meets the "
                 f"minimum of {inr(MIN_MONTHLY_INCOME)}."
                 if income >= MIN_MONTHLY_INCOME else
                 f"Household income of {inr(income)}/month is below the "
                 f"minimum of {inr(MIN_MONTHLY_INCOME)}."),
    })

    unemployed_alone = (values["employment"] == 2
                        and values["coapplicant_income"] == 0)
    checks.append({
        "name": "Source of income",
        "passed": not unemployed_alone,
        "text": ("An unemployed applicant needs a co-applicant with income."
                 if unemployed_alone else
                 "There is an income source to repay the loan."),
    })

    ratio = foir(values)
    new_emi = emi(values["loan_amount"], values["loan_term"])
    checks.append({
        "name": "Affordability (EMI to income)",
        "passed": ratio <= MAX_FOIR,
        "text": (f"All EMIs, including about {inr(new_emi)}/month for this "
                 f"loan, would take {ratio * 100:.0f}% of household income "
                 f"(limit {MAX_FOIR * 100:.0f}%)."),
    })

    age = values["age"]
    age_end = age + values["loan_term"] / 12
    age_ok = MIN_AGE <= age <= MAX_AGE
    checks.append({
        "name": "Age",
        "passed": age_ok and age_end <= MAX_AGE_AT_MATURITY,
        "text": (f"Applicants must be {MIN_AGE}–{MAX_AGE} years old; "
                 f"you are {age}." if not age_ok else
                 f"You would be {age_end:.0f} when the loan ends, above the "
                 f"limit of {MAX_AGE_AT_MATURITY}."
                 if age_end > MAX_AGE_AT_MATURITY else
                 f"Age {age} is eligible and the loan ends by age "
                 f"{age_end:.0f}."),
    })

    return checks
