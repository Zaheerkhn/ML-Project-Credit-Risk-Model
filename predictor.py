from joblib import load
import pandas as pd
import numpy as np

model_data = load('Artifacts/model_data.joblib')
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']

def create_input_dataframe(age, income, loan_amount, loan_tenure_months, number_of_open_accounts, 
                            credit_utilization_ratio, del_months_to_loan_months, avg_dpd_per_delinquency, 
                            residence_type, loan_purpose, loan_type):
    # Calculate loan_to_income_ratio
    loan_to_income_ratio = loan_amount / income if income != 0 else 0
    
    # Create a dictionary with all the inputs
    input_dict = {
        'age': [age],
        'loan_tenure_months': [loan_tenure_months],
        'number_of_open_accounts': [number_of_open_accounts],
        'credit_utilization_ratio': [credit_utilization_ratio],
        'loan_to_income_ratio': [loan_to_income_ratio],
        'del_months_to_loan_months': [del_months_to_loan_months],
        'avg_dpd_per_delinquency': [avg_dpd_per_delinquency],
        'residence_type_Owned': [1 if residence_type == 'Owned' else 0],
        'residence_type_Mortgage': [1 if residence_type == 'Mortgage' else 0],
        'residence_type_Rented': [1 if residence_type == 'Rented' else 0],
        'loan_purpose_Education': [1 if loan_purpose == 'Education' else 0],
        'loan_purpose_Home': [1 if loan_purpose == 'Home' else 0],
        'loan_purpose_Auto': [1 if loan_purpose == 'Auto' else 0],
        'loan_purpose_Personal': [1 if loan_purpose == 'Personal' else 0],
        'loan_type_Secured': [1 if loan_type == 'Secured' else 0],
        'loan_type_Unsecured': [1 if loan_type == 'Unsecured' else 0],
        # additional dummy fields just for scaling purpose
        'number_of_dependants': 1,  # Dummy value
        'years_at_current_address': 1,  # Dummy value
        'zipcode': 1,  # Dummy value
        'sanction_amount': 1,  # Dummy value
        'processing_fee': 1,  # Dummy value
        'gst': 1,  # Dummy value
        'net_disbursement': 1,  # Computed dummy value
        'principal_outstanding': 1,  # Dummy value
        'bank_balance_at_application': 1,  # Dummy value
        'number_of_closed_accounts': 1,  # Dummy value
        'enquiry_count': 1  # Dummy value
    }

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame(input_dict)
    input_df[cols_to_scale] =scaler.transform(input_df[cols_to_scale])
    input_df = input_df[features]
    
    return input_df

def predict(age, income, loan_amount, loan_tenure_months, number_of_open_accounts, 
                            credit_utilization_ratio, del_months_to_loan_months, avg_dpd_per_delinquency, 
                            residence_type, loan_purpose, loan_type):
    
    input_df = create_input_dataframe(age, income, loan_amount, loan_tenure_months, number_of_open_accounts, 
                            credit_utilization_ratio, del_months_to_loan_months, avg_dpd_per_delinquency, 
                            residence_type, loan_purpose, loan_type)
    
    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Apply the logistic function to calculate the probability
    default_probability = 1 / (1 + np.exp(-x))

    non_default_probability = 1 - default_probability

    # Convert the probability to a credit score, scaled to fit within 300 to 900
    credit_score = base_score + non_default_probability.flatten() * scale_length

    # Determine the rating category based on the credit score
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'  # in case of any unexpected score

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating