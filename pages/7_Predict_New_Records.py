import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(page_title="Step 7 - Predict New Record", layout="wide")
st.title("🔮 Step 7: Predict Using Manual Input")

# -------------------- Load model and features --------------------
model_path = "models/logistic_model.pkl"
feature_path = "models/feature_list.pkl"

if not os.path.exists(model_path) or not os.path.exists(feature_path):
    st.error("❌ Model or feature list not found. Please complete training and export first.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(feature_path, "rb") as f:
    feature_list = pickle.load(f)

# -------------------- Form Input --------------------
st.subheader("📋 Enter Feature Values Manually")

input_data = {}

with st.form("prediction_form"):
    for feature in feature_list:
        value = st.text_input(f"{feature}", key=f"input_{feature}")
        input_data[feature] = value
    
    submitted = st.form_submit_button("Predict")

# -------------------- Predict Probability --------------------
if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df.apply(pd.to_numeric, errors="coerce")

        if input_df.isnull().any().any():
            st.error("❌ Please enter valid numeric values for all fields.")
        else:
            prob = model.predict_proba(input_df)[0][1]  # Probability of class 1
            predicted_class = 1 if prob >= 0.5 else 0

            st.success(f"✅ Predicted Probability of class **1**: `{prob:.4f}`")
            st.info(f"🔍 Final Prediction (based on threshold 0.5): `{predicted_class}`")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
