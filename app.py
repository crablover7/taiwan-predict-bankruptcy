import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    file_path = "filtered_data.csv"  # Make sure this file is in your GitHub repo
    df = pd.read_csv(file_path)
    return df

# Train XGBoost model
@st.cache_resource
def train_model(df):
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1].astype(int)  # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, X.columns

# Streamlit UI
st.title("Bankruptcy Prediction App 🚀")

df = load_data()
model, accuracy, feature_names = train_model(df)

st.write(f"### Model Accuracy: {accuracy:.4f}")

# User Input Section
st.write("### Enter Company Financial Data:")
user_input = []

for feature in feature_names:
    value = st.number_input(f"{feature}", value=float(df[feature].mean()))
    user_input.append(value)

# Prediction Button
if st.button("Predict Bankruptcy"):
    prediction = model.predict([user_input])
    result = "🔴 Bankrupt" if prediction[0] == 1 else "🟢 Not Bankrupt"
    st.subheader(f"Prediction: {result}")
