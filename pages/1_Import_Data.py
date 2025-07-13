import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Step 1 - Import Data", layout="wide")
st.title("📂 Step 1: Import Dataset")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Store in session state
    st.session_state["raw_data"] = df

    # Save to disk
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/uploaded_data.csv", index=False)

    st.success("✅ Data uploaded and saved successfully!")
    
    # Show preview
    st.subheader("📊 Preview of the Uploaded Data")
    st.dataframe(df.head())

else:
    st.warning("⚠️ Please upload a CSV file to continue.")
