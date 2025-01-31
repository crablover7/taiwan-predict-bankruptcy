import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("bankruptcy_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define features
selected_feature_names = [
    "Persistent EPS in the Last Four Seasons",
    "Non-industry income and expenditure/revenue",
    "Borrowing dependency",
    "Total debt/Total net worth",
    "Net Income to Total Assets",
    "Current Liability to Assets",
    "Net worth/Assets",
    "Quick Ratio",
    "ROA(C) before interest and depreciation before interest",
    "ROA(B) before interest and depreciation after tax",
    "Equity to Liability",
    "Net Income to Stockholder's Equity",
    "Revenue Per Share (Yuan ¥)",
    "Retained Earnings to Total Assets",
    "Operating Profit Rate",
    "Degree of Financial Leverage (DFL)",
    "ROA(A) before interest and % after tax",
    "Debt ratio %",
    "Accounts Receivable Turnover",
    "Net Value Per Share (C)"
]

# Streamlit UI
st.title("Bankruptcy Prediction App")
st.write("Enter financial details below to predict if a company is at risk of bankruptcy.")

# User input form
user_input = []
for feature in selected_feature_names:
    value = st.number_input(f"{feature}:", value=0.0, format="%.10f")
    user_input.append(value)

input_data = np.array(user_input).reshape(1, -1)

# Predict Button
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of bankruptcy

    if prediction[0] == 1:
        st.error(f"⚠️ The company is **at risk of bankruptcy** with probability {probability:.10f}.")
    else:
        st.success(f"✅ The company is **not at risk of bankruptcy** with probability {1 - probability:.10f}.")
