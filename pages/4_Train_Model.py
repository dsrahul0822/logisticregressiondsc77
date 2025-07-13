import streamlit as st
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Step 4 - Train Model", layout="wide")
st.title("🤖 Step 4: Train Logistic Regression Model")

# -------------------- Validate session --------------------
if "X_train" not in st.session_state or "y_train" not in st.session_state:
    st.warning("⚠️ Please complete Step 3: Train-Test Split first.")
    st.stop()

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]

st.subheader("📊 Training Data Summary")
st.write("🔹 X_train shape:", X_train.shape)
st.write("🔹 y_train shape:", y_train.shape)

# -------------------- Train the model --------------------
if st.button("🚀 Train Logistic Regression Model"):
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Save model and feature list to session state
        st.session_state["trained_model"] = model
        st.session_state["feature_list"] = X_train.columns.tolist()

        st.success("✅ Model trained successfully!")

        # Show coefficients
        st.subheader("📈 Model Coefficients")
        coef_df = {
            "Feature": X_train.columns,
            "Coefficient": model.coef_[0]
        }
        st.dataframe(coef_df)

        st.markdown(f"**📍 Model Intercept:** `{model.intercept_[0]:.4f}`")

    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
