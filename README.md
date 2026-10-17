# LoanLens — Loan Approval Prediction

LoanLens predicts whether a loan application is likely to be approved, from the applicant's financial and demographic details. It covers the whole machine learning workflow: data checks, cleaning, encoding, model training and comparison, evaluation, and a Flask web app that returns a prediction in real time.

**Live demo:** https://loan-lens-black.vercel.app/

## How it works

1. The applicant fills in a form: gender, marital status, dependents, education, employment type, property area, credit score, applicant and co-applicant income, loan amount and loan term.
2. The Flask backend turns the form into an 11-value feature vector, in the same order used for training.
3. A trained **logistic regression** model returns the probability of approval.
4. Applications with a probability of **0.40 or higher** are shown as approved; the rest as rejected.
5. `explain.py` explains the decision. Because the model is a scaled logistic regression, each input's contribution to the score is exactly `coefficient × (value − training mean) / training std`, the same as its SHAP value for a linear model. The result page shows:
   - the approval probability,
   - the top reasons for the decision in plain language (e.g. "Your credit score of 480 is below the typical applicant's score of about 617"),
   - the strongest factors that pushed the other way,
   - for rejections, what would change the decision (e.g. the credit score or loan amount needed, everything else unchanged),
   - a bar breakdown of every factor's effect.

   Gender and marital status appear in the breakdown for transparency but are never given as reasons.

## Model

| Item | Detail |
|---|---|
| Dataset | `train_with_credit_score.xlsx`, 3,192 applications, 50% approved |
| Features (11) | Gender, Married, Dependents, Education, Applicant income, Co-applicant income, Loan amount, Loan term, Property area, Credit score, Self-employed |
| Dropped columns | `Loan_ID`, `Age`, `Credit_History` (identifier or noisy/leaky) |
| Pipeline | `StandardScaler` → `LogisticRegression(C=0.3, class_weight="balanced")` |
| Split | 70% train / 30% test, stratified, `random_state=42` |
| Decision threshold | 0.40 |

Performance of the saved model (`loan_model.pkl`) on the 30% test split:

| Metric | Value |
|---|---|
| ROC-AUC | 0.985 |
| Accuracy (threshold 0.40) | 94.8% |
| Precision | 91.2% |
| Recall | 99.2% |
| F1-score | 0.95 |

A random forest was also tried for comparison (`day7_model_comparison.py`); logistic regression was kept because it is simple, fast and easy to explain.

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
├── explain.py                    Turns the model's weights into reasons for each decision
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

The `day*` scripts record the project's development step by step; the app only needs `app.py`, `explain.py`, `loan_model.pkl`, `templates/` and `static/`.

## Limitations

- Predictions are only as good as the dataset, and should support, not replace, a loan officer's judgement.
- Gender and marital status are model inputs. In a real lending system they would need a fairness review, since lending decisions should not depend on them.
