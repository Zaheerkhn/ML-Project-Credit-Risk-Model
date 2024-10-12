import streamlit as st 
from predictor import predict
# Define the Streamlit app UI
st.title('Credit Risk Prediction')

# Create a 4x3 grid for input features
row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2, row2_col3 = st.columns(3)
row3_col1, row3_col2, row3_col3 = st.columns(3)
row4_col1, row4_col2, row4_col3 = st.columns(3)

# Collect basic user inputs in the first row
with row1_col1:
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
with row1_col2:
    income = st.number_input('Income', min_value=0, value=0)
with row1_col3:
    loan_amount = st.number_input('Loan Amount', min_value=0, value=0)

# Calculate loan to income ratio and display in the first column of the second row
loan_to_income_ratio = loan_amount / income if income != 0 else 0
with row2_col1:
    st.write(f'Loan to Income Ratio: {loan_to_income_ratio:.2f}')

# Collect additional user inputs in the subsequent rows
with row2_col2:
    loan_tenure_months = st.number_input('Loan Tenure (Months)', min_value=0, value=1)
with row2_col3:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Mortgage', 'Rented'])

with row3_col1:
    number_of_open_accounts = st.number_input('Number of Open Accounts', min_value=0, max_value=50, value=0)
with row3_col2:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio (%)', min_value=1, max_value=100, value=1)
with row3_col3:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])

with row4_col1:
    loan_type = st.selectbox('Loan Type', ['Secured', 'Unsecured'])
with row4_col2:
    del_months_to_loan_months = st.number_input('Delinquency Months to Loan Months', min_value=0, max_value=100, step=1, value=30)
with row4_col3:
    avg_dpd_per_delinquency = st.number_input('Average DPD per Delinquency', min_value=0, value=20)

# Predict credit risk
if st.button('Predict Credit Risk'):
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months, number_of_open_accounts, 
                            credit_utilization_ratio, del_months_to_loan_months, avg_dpd_per_delinquency, 
                            residence_type, loan_purpose, loan_type)
    
    st.write(f"Deafult Probability: {probability:.2%}")
    st.write(f"Credit Score: {credit_score}")
    st.write(f"Rating: {rating}")