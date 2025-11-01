import streamlit as st
import joblib

# Load model and scaler
model = joblib.load("emi_predict_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💰 EMI Predict AI - Financial Risk Assessment App")
st.write("Predict EMI eligibility using Machine Learning")

# === INPUT FIELDS ===
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income (₹)", min_value=1000, value=30000)
loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, value=100000)

# === SCALE & PREDICT ===
if st.button("Predict EMI Eligibility"):
    try:
        # Scale only 3 features (age, income, loan_amount)
        X = scaler.transform([[age, income, loan_amount]])
        prediction = model.predict(X)[0]

        if prediction == 1:
            st.success("✅ Eligible for EMI!")
        else:
            st.error("❌ Not eligible for EMI.")
    except Exception as e:
        st.error(f"Error: {e}")
