import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load version-safe models
model = joblib.load('best_delay_model_v2.joblib')
feature_cols = joblib.load('feature_cols_v2.joblib')

st.title("📦 Predictive Delivery Optimizer")
st.write("Estimate delay probability for a given shipment.")

# Inputs
Promised_Delivery_Days = st.number_input("Promised Delivery Days", 0, 30, 3)
Distance_KM = st.number_input("Distance (KM)", 0.0, 2000.0, 100.0)
Order_Value_INR = st.number_input("Order Value (INR)", 0.0, 50000.0, 1000.0)
Traffic_Delay_Minutes = st.number_input("Traffic Delay (mins)", 0.0, 300.0, 0.0)
is_express = st.checkbox("Express Delivery?")
weather_issue = st.checkbox("Weather Issue?")

# Predict button
if st.button("Predict Delay Risk"):
    input_df = pd.DataFrame([[
        Promised_Delivery_Days, Distance_KM, Order_Value_INR,
        Traffic_Delay_Minutes, int(is_express), int(weather_issue)
    ]], columns=feature_cols)

    prob = model.predict_proba(input_df)[0][1]
    st.metric("Predicted Delay Probability", f"{prob*100:.1f}%")
    if prob >= 0.7:
        st.error("⚠️ High risk — use express or reroute.")
    elif prob >= 0.4:
        st.warning("🕓 Moderate risk — monitor closely.")
    else:
        st.success("✅ Low risk — delivery on time.")
