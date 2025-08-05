import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Employee Salary Prediction")

# Create 3 columns for integer input
col1, col2, col3 = st.columns(3)

with col1:
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, step=1)

with col2:
    satisfaction_level = st.number_input("Satisfaction Level (0-100)", min_value=0, max_value=100, step=1)

with col3:
    avg_monthly_hours = st.number_input("Average Monthly Hours", min_value=0, max_value=400, step=1)

# Convert satisfaction level from 0-100 to 0.0-1.0 (if needed)
satisfaction_level_scaled = satisfaction_level / 100

if st.button("Predict Salary"):
    input_data = np.array([[years_at_company, satisfaction_level_scaled, avg_monthly_hours]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
