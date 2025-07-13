import streamlit as st
import pickle
import os

st.set_page_config(page_title="Step 6 - Export Model", layout="wide")
st.title("💾 Step 6: Export Trained Model and Features")

# -------------------- Check Dependencies --------------------
if "trained_model" not in st.session_state or "feature_list" not in st.session_state:
    st.warning("⚠️ Please train the model first (Step 4).")
    st.stop()

model = st.session_state["trained_model"]
features = st.session_state["feature_list"]

# -------------------- Save Files --------------------
os.makedirs("models", exist_ok=True)

model_path = "models/logistic_model.pkl"
feature_path = "models/feature_list.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(feature_path, "wb") as f:
    pickle.dump(features, f)

st.success("✅ Model and feature list saved successfully!")

# -------------------- Download Buttons --------------------
st.download_button(
    label="📥 Download Trained Model (.pkl)",
    data=open(model_path, "rb").read(),
    file_name="logistic_model.pkl"
)

st.download_button(
    label="📥 Download Feature List (.pkl)",
    data=open(feature_path, "rb").read(),
    file_name="feature_list.pkl"
)
