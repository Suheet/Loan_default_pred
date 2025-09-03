import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("logr_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📊 Loan Default Prediction App")
st.write("Enter applicant details to predict the probability of loan default.")

# Input fields
education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
employment = st.selectbox("Employment Type", ["Unemployed", "Part-time", "Full-time", "Self-employed"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
mortgage = st.radio("Has Mortgage?", ["Yes", "No"])
dependents = st.radio("Has Dependents?", ["Yes", "No"])
purpose = st.selectbox("Loan Purpose", ["Other", "Auto", "Business", "Home", "Education"])
cosigner = st.radio("Has Co-signer?", ["Yes", "No"])

# Mappings (same as you used in training)
education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
employment_map = {"Unemployed": 0, "Part-time": 1, "Full-time": 2, "Self-employed": 3}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
yes_no_map = {"No": 0, "Yes": 1}
loan_purpose_map = {"Other": 0, "Auto": 1, "Business": 2, "Home": 3, "Education": 4}

# Convert inputs to numeric
features = np.array([[
    education_map[education],
    employment_map[employment],
    marital_map[marital],
    yes_no_map[mortgage],
    yes_no_map[dependents],
    loan_purpose_map[purpose],
    yes_no_map[cosigner]
]])

# Prediction
if st.button("🔮 Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1]

    if prediction == 1:
        st.error(f"⚠️ Loan will **Default** (Prob: {probability:.2f})")
    else:
        st.success(f"✅ Loan will **Not Default** (Prob: {probability:.2f})")

