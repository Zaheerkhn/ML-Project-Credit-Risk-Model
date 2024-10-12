# Credit Risk Prediction Model

This project implements a Credit Risk Prediction model using logistic regression to assess the probability of default on loans. The model is designed to help financial institutions evaluate credit risk based on various features provided by applicants.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Evaluation](#model-evaluation)

## Features
- User-friendly interface built with Streamlit for inputting applicant details.
- Predicts the probability of loan default based on various features:
  - Age
  - Loan Tenure (Months)
  - Number of Open Accounts
  - Credit Utilization Ratio
  - Loan to Income Ratio
  - Delinquency Ratio
  - Average DPD per Delinquency
  - Residence Type (Owned, Mortgage, Rented)
  - Loan Purpose (Education, Home, Auto, Personal)
  - Loan Type (Secured, Unsecured)

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- XGBoost (optional for experimentation)
- Pandas
- NumPy
- Joblib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Zaheerkhn/ML-Project-Credit-Risk-Model.git
   cd Credit-Risk-Prediction

2. **Create a virtual environment (optional but recommended)**:
    ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install the required packages**:
    ```bash
   pip install -r requirements.txt

4. **Run the Streamlit app**:
   ```bash
   streamlit run main.py

## How It Works
1. The user inputs their details into the Streamlit interface.
2. The application calculates the Loan to Income Ratio based on the provided income and loan amount.
3. Inputs are transformed and passed to the trained logistic regression model to predict the probability of default.
4. The results, including default probability, credit score, and rating, are displayed to the user.

## Model Evaluation
1. The model's performance is evaluated using metrics such as the F1 Score and KS Statistic.
2. The model's ability to distinguish between default and non-default cases is visualized using ROC and AUC curves.
