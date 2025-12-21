# from flask import Flask, render_template, request
# import joblib

# app = Flask(__name__)

# # Load saved modelcls
# AI_model = joblib.load('ans.pkl')
# @app.route('/')
# def home():
#     return render_template('index.html')
# @app.route('/predict', methods=['POST'])
# def predict():
#     age = int(request.form['age'])
#     salary = int(request.form['salary'])
#     prediction = AI_model.predict([[age, salary]])
#     if prediction[0] == 0:
#         result = "PERSON DID NOT PURCHASE"
#     else:
#         result = "PERSON PURCHASED"
#     return render_template('index.html', result=result)
# if __name__ == '__main__':
#     app.run(debug=True)
import streamlit as st
import joblib
import numpy as np

# Load trained model
AI_model = joblib.load("ans.pkl")

# App title
st.title("🛒 Purchase Prediction App")

st.write("Predict whether a person will purchase or not")

# User inputs
age = st.number_input("Enter Age", min_value=18, max_value=100, value=25)
salary = st.number_input("Enter Salary", min_value=10000, max_value=200000, value=50000)

# Predict button
if st.button("Predict"):
    prediction = AI_model.predict([[age, salary]])

    if prediction[0] == 0:
        st.error("❌ PERSON DID NOT PURCHASE")
    else:
        st.success("✅ PERSON PURCHASED")


