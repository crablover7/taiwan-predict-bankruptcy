import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("xgb_bankruptcy_model.pkl")

st.title("📊 Bankruptcy Prediction Dashboard")
st.markdown("Enter financial details below to predict the risk of bankruptcy.")

# Input fields
feature_1 = st.number_input("Enter feature 1 value:", value=0.0)
feature_2 = st.number_input("Enter feature 2 value:", value=0.0)
feature_3 = st.number_input("Enter feature 3 value:", value=0.0)
feature_4 = st.number_input("Enter feature 4 value:", value=0.0)
feature_5 = st.number_input("Enter feature 5 value:", value=0.0)

# Convert inputs to numpy array
input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])

# Predict on user input
if st.button("Predict Bankruptcy Risk"):
    y_pred_prob = model.predict_proba(input_data)[:, 1]  # Get probability
    threshold = 0.3
    y_pred = (y_pred_prob > threshold).astype(int)  # Apply threshold
    
    if y_pred[0] == 1:
        st.error(f"⚠️ High Risk of Bankruptcy! (Probability: {y_pred_prob[0]:.2f})")
    else:
        st.success(f"✅ Low Risk of Bankruptcy! (Probability: {y_pred_prob[0]:.2f})")
