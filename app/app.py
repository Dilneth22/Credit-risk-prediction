import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os # Import the os library

# --- ROBUST PATHING ---
# This makes the app aware of its own location and builds absolute paths
# This is the key change to fix the "file not found" error
APP_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(APP_DIR, 'scaler.pkl')
COLUMNS_PATH = os.path.join(APP_DIR, 'model_columns.pkl')
# --- END OF ROBUST PATHING ---

# Load the trained model, scaler, and columns using the absolute paths
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
except FileNotFoundError:
    st.error("Model files not found! The app is looking for them at these locations:")
    st.write(f"Model: {MODEL_PATH}")
    st.write(f"Scaler: {SCALER_PATH}")
    st.write(f"Columns: {COLUMNS_PATH}")
    st.write("Please ensure these files exist after running the Jupyter Notebook.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()


st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("Loan Approval Prediction")
st.write("Enter the applicant's details to predict their loan approval status.")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=1500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=150.0)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=12.0, value=360.0, step=12.0)
    credit_history = st.selectbox("Credit History Available", [1.0, 0.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])


if st.button("Predict Loan Status"):
    # Create a dictionary from the inputs
    input_data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Married_Yes': 1 if married == 'Yes' else 0,
        'Dependents_1': 1 if dependents == '1' else 0,
        'Dependents_2': 1 if dependents == '2' else 0,
        'Dependents_3+': 1 if dependents == '3+' else 0,
        'Education_Not Graduate': 1 if education == 'Not Graduate' else 0,
        'Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,
        'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
        'Property_Area_Urban': 1 if property_area == 'Urban' else 0,
    }

    # Convert to DataFrame and align columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"**Approved**")
        st.progress(prediction_proba[0][1])
        st.metric(label="Approval Probability", value=f"{prediction_proba[0][1]:.2%}")
    else:
        st.error(f"**Not Approved**")
        st.progress(prediction_proba[0][0])
        st.metric(label="Rejection Probability", value=f"{prediction_proba[0][0]:.2%}")

