import streamlit as st
import pickle
import numpy as np

# load model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🛒 Social Network Ads Prediction")
st.write("Predict whether a user will purchase or not")

# user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Estimated Salary", min_value=1000, max_value=200000, value=87000)

if st.button("Predict"):
    # prepare input
    new_data = np.array([[age, salary]])
    new_data_scaled = scaler.transform(new_data)

    # prediction
    prediction = model.predict(new_data_scaled)

    if prediction[0] == 1:
        st.success("✅ Prediction: Purchased")
    else:
        st.warning("❌ Prediction: Not Purchased")
