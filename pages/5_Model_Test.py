import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Step 5 - Model Testing", layout="wide")
st.title("🧪 Step 5: Test & Evaluate Model")

# -------------------- Load Data --------------------
if "trained_model" not in st.session_state:
    st.warning("⚠️ Please train the model in Step 4 first.")
    st.stop()

model = st.session_state["trained_model"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

# -------------------- Perform Prediction --------------------
st.subheader("🔍 Making Predictions on Test Set")

y_pred = model.predict(X_test)

# -------------------- Metrics --------------------
st.subheader("📊 Evaluation Metrics")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

st.write(f"✅ **Accuracy:** `{acc:.4f}`")
st.write(f"✅ **Precision:** `{prec:.4f}`")
st.write(f"✅ **Recall:** `{rec:.4f}`")
st.write(f"✅ **F1 Score:** `{f1:.4f}`")

# -------------------- Confusion Matrix --------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

# -------------------- Classification Report --------------------
st.subheader("🧾 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())
