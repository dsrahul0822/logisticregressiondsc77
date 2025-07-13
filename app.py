import streamlit as st

# Set config and title
st.set_page_config(page_title="Logistic Regression App", layout="wide")
st.title("🔍 Logistic Regression Solver")

st.markdown("""
Welcome to the **Logistic Regression App**!  
This application will walk you through solving a binary classification problem step-by-step using:
- Data Import
- Preprocessing (Missing values, Outliers, Encoding)
- Train/Test Split
- Model Training
- Model Evaluation
- Model Export

👉 Use the sidebar to navigate through each step.
""")
