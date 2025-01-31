import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("xgb_bankruptcy_model.pkl")

st.title("📊 Bankruptcy Prediction Dashboard")
st.markdown("Enter financial details below to predict the risk of bankruptcy.")

# Feature names from the image
feature_names = [
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

# Create input fields dynamically
user_inputs = []
for feature in feature_names:
    value = st.number_input(f"{feature}:", value=0.0)
    user_inputs.append(value)

# Convert inputs to numpy array
input_data = np.array([user_inputs]).astype(float)

# Predict on user input
if st.button("Predict Bankruptcy Risk"):
    y_pred_prob = model.predict_proba(input_data)[:, 1]  # Get probability
    threshold = 0.3
    y_pred = (y_pred_prob > threshold).astype(int)  # Apply threshold
    
    if y_pred[0] == 1:
        st.error(f"⚠️ High Risk of Bankruptcy! (Probability: {y_pred_prob[0]:.2f})")
    else:
        st.success(f"✅ Low Risk of Bankruptcy! (Probability: {y_pred_prob[0]:.2f})")
