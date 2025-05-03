import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("app/shrinkage_model.h5")
scaler = joblib.load("app/scaler.pkl")

# Get feature names from scaler to ensure exact order
feature_names = scaler.feature_names_in_

# Streamlit app title
st.title("🧵 Shrinkage Prediction App")
st.markdown("Enter the fabric properties below to predict shrinkage after finishing.")

# Collect user inputs
user_inputs = []
for feature in feature_names:
    val = st.number_input(f"{feature}", format="%.4f")
    user_inputs.append(val)

# Predict on button click
if st.button("Predict Shrinkage"):
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([user_inputs], columns=feature_names)

    # Scale inputs
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)

    # Show results
    st.subheader("📉 Predicted Shrinkage:")
    st.write(f"**Length Shrinkage (-6.5%)**: {prediction[0][0]:.4f}")
    st.write(f"**Width Shrinkage (-6.5%)**:  {prediction[0][1]:.4f}")
