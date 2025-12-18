import streamlit as st
import os
import pickle
import numpy as np

# -----------------------------
# Set up paths and load files
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "knn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛒 Social Network Ads Prediction")
st.write("Predict whether a user will purchase or not")

# Example: user input
age = st.number_input("Enter Age")
salary = st.number_input("Enter Estimated Salary")

# Scale input and make prediction
input_data = scaler.transform([[age, salary]])
prediction = model.predict(input_data)

st.write("Prediction:", "Purchased" if prediction[0] == 1 else "Not Purchased")
