import streamlit as st
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "ans.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.title("🛒 Purchase Prediction App")

age = st.number_input("Enter Age", 18, 100, 25)
salary = st.number_input("Enter Salary", 10000, 200000, 50000)

if st.button("Predict"):
    data = scaler.transform([[age, salary]])
    pred = model.predict(data)

    if pred[0] == 0:
        st.error("❌ PERSON DID NOT PURCHASE")
    else:
        st.success("✅ PERSON PURCHASED")
