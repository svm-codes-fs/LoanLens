# LoanLens — Loan Approval Prediction

LoanLens predicts whether a loan application is likely to be approved, from the applicant's financial and demographic details. It covers the whole machine learning workflow: data checks, cleaning, encoding, model training and comparison, evaluation, and a Flask web app that returns a prediction in real time.

**Live demo:** https://loan-lens-black.vercel.app/

## How it works

LoanLens works like a real lender: hard **policy checks** first, then a **risk model**, then a plain-language **explanation**. A loan is approved only if it passes every check *and* the model gives it at least a 50% approval probability.

1. The applicant fills in a form: age, dependents, education, employment (salaried / self-employed / unemployed), property area, credit score, applicant and co-applicant income, existing EMIs, loan amount and loan term.
2. **Bank policy checks** (`policy.py`), using limits commonly quoted by Indian lenders:

   | Check | Rule |
   |---|---|
   | Minimum credit score | 650 or higher |
   | Minimum income | Household income of at least ₹15,000/month |
   | Source of income | An unemployed applicant needs a co-applicant with income |
   | Affordability (FOIR) | Existing EMIs + this loan's EMI ≤ 50% of household income (EMI estimated at 10.5% p.a.) |
   | Age | 21–60 when applying, and the loan must end by age 65 |

3. **Risk model** (`features.py`, `loan_model.pkl`): a logistic regression trained on the dataset, using credit-score bands, household income, dependents, education, employment and property area.
4. **Explanation** (`explain.py`). Because the model is a scaled logistic regression, each input's contribution is exactly `coefficient × (value − training mean) / training std`, the same as its SHAP value for a linear model. The result page shows:
   - the decision and the model's approval probability,
   - why: any failed policy check first, then the strongest model factors, in plain language,
   - the strongest factors that pushed the other way,
   - for rejections, what single change would get the loan approved (credit score, loan amount, term or co-applicant income), found by re-running the full decision,
   - every policy check with ✔ / ✘, and a bar breakdown of every model factor.

## Model

| Item | Detail |
|---|---|
| Dataset | `train_with_credit_score.xlsx`, 3,192 applications, 50% approved |
| Features (9) | Credit score ≥ 600 / ≥ 650 / ≥ 700, log household income (clipped to ₹15k–₹2L), dependents, graduate, self-employed, unemployed, property area |
| Not used | `Loan_ID` (identifier), `Credit_History` (leaks the label), `Age` (policy check instead), `Gender` and `Married` (lending decisions should not depend on them), loan amount / EMI burden (policy check instead, see below) |
| Pipeline | `StandardScaler` → `LogisticRegression` |
| Decision threshold | 0.50 |

Performance (`python train_model.py`):

| Metric | Value |
|---|---|
| 5-fold CV ROC-AUC | 0.999 |
| 5-fold CV accuracy | 98.3% |
| Hold-out accuracy (30%) | 98.1% |
| Hold-out precision | 96.6% |
| Hold-out recall | 99.8% |

### Why the model changed

The first model (raw inputs, threshold 0.40, 94.8% accuracy) rejected many strong applications:

- **It judged the loan amount in rupees, not affordability.** Training loans are all under ₹3.7 lakh, but the form allows up to ₹50 lakh, so the straight-line penalty went far outside anything it had seen. A ₹20 lakh loan was rejected even with an 800 credit score and ₹2.5 lakh/month income, while a 500 score with high income was approved.
- **Credit score was a straight line**, but in the data approval jumps at 650 (under 5% approved below it, about 97% above). Credit-score bands capture that, and match how banks treat CIBIL scores.
- **Encoding bugs:** "Unemployed" was not in the mapping and silently became "salaried"; missing incomes and loan amounts were filled with 0.

Every loan in the dataset costs under 20% of income (about 3% on average), and approval rates don't change with it, so the data cannot teach affordability. That is why affordability is a policy check rather than a model feature.

## Tech stack

Python · pandas · NumPy · scikit-learn · Flask · HTML · CSS · JavaScript · Vercel

## Getting started

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Then open http://127.0.0.1:5000.

To re-run the training scripts you also need pandas and openpyxl (for the Excel files):

```bash
pip install pandas openpyxl
```

```bash
python train_model.py
```

`train_model.py` overwrites `loan_model.pkl` with a freshly trained model.

## Project structure

```
├── app.py                        Flask app: form page and /predict route
├── features.py                   Feature engineering shared by training and the app
├── policy.py                     Bank policy checks (credit score, income, FOIR, age)
├── explain.py                    Decision (checks + model) and its explanation
├── train_model.py                Final training pipeline; saves loan_model.pkl
├── loan_model.pkl                Trained model used by the app
├── train_with_credit_score.xlsx  Raw dataset
├── Day2_data_check.py            Step 1: shape, types, missing values, statistics
├── day3_fix_missing.py           Step 2: fill missing values  → loan_cleaned.xlsx
├── day4_encoding.py              Step 3: encode categories   → loan_encoded.csv
├── day5_first_model.py           Step 4: first logistic regression
├── day6_model_evaluation.py      Step 5: confusion matrix and classification report
├── day7_model_comparison.py      Step 6: logistic regression vs random forest
├── templates/index.html          The form and result page
├── static/                       CSS, form script and images
├── requirements.txt              Packages needed to run the app
└── vercel.json                   Vercel deployment settings
```

The `day*` scripts record the project's development step by step; the app only needs `app.py`, `features.py`, `policy.py`, `explain.py`, `loan_model.pkl`, `templates/` and `static/`.

## Limitations

- The dataset looks synthetic: approval is decided almost entirely by a credit-score cutoff at 650, and loans are small relative to income. Real bank data would include repayment history, existing debts, employment length and defaults.
- The policy limits are indicative values commonly quoted for Indian personal loans, not any one bank's actual policy.
- Predictions should support, not replace, a loan officer's judgement.
