LoanLens — Loan Approval Prediction System

LoanLens is a Machine Learning-based web application designed to predict loan approval outcomes using applicant financial and demographic information. The project demonstrates an end-to-end Machine Learning workflow, from data preprocessing and model training to web-based model deployment.

The system uses Python, Pandas, NumPy, and Scikit-learn for data analysis and predictive modeling. Applicant data such as income, education, employment status, loan amount, loan term, marital status, property area, and credit history is processed and transformed into a suitable format for Machine Learning. Missing values are handled, categorical features are encoded, and the dataset is prepared for model training and evaluation.

A Logistic Regression classification model is used to predict whether a loan application is likely to be approved or rejected. The model is evaluated using performance metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC, achieving an AUC score of approximately 0.94.

To make the trained model accessible to users, the project integrates the Machine Learning model with a Flask backend. A user-friendly web interface built using HTML and CSS allows users to enter applicant details and receive a real-time loan approval prediction.

Key Features
Loan approval prediction using Machine Learning
Data preprocessing and cleaning
Missing value handling
Categorical feature encoding
Feature engineering and model training
Logistic Regression classification
Model performance evaluation
Flask-based web application
Real-time prediction
Interactive HTML/CSS user interface
Serialized Machine Learning model using Pickle
Tech Stack

Python | Pandas | NumPy | Scikit-learn | Flask | HTML | CSS | Pickle

The project showcases practical skills in Machine Learning, Data Analysis, Predictive Modeling, Feature Engineering, Classification, Model Evaluation, Python Development, Flask, and Model Deployment, while demonstrating how predictive analytics can be applied to real-world financial decision-making.
