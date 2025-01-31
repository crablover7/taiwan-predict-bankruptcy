import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Bankruptcy Prediction with XGBoost")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write("### Model Accuracy")
    st.success(f"{accuracy:.4f}")

    st.write("### Make a Prediction")
    input_data = []
    for col in X.columns:
        value = st.number_input(f"{col}", value=float(X[col].mean()))
        input_data.append(value)

    if st.button("Predict"):
        prediction = model.predict([input_data])
        st.write(f"**Prediction:** {'Bankrupt' if prediction[0] == 1 else 'Not Bankrupt'}")
