import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Step 3 - Train/Test Split", layout="wide")
st.title("✂️ Step 3: Train-Test Split")

# -------------------- Load Processed Data --------------------
if "processed_data" not in st.session_state:
    st.warning("⚠️ Please complete Step 2: Data Preprocessing first.")
    st.stop()

df = st.session_state["processed_data"].copy()

st.subheader("📊 Current Processed Data")
st.dataframe(df.head())

# -------------------- Target Column Selection --------------------
st.subheader("🎯 Select Target Column")
target_col = st.selectbox("Choose the target column (y):", df.columns)

# -------------------- Feature Column Selection --------------------
st.subheader("📥 Select Feature Columns (X)")
available_features = df.columns.drop(target_col)
selected_features = st.multiselect(
    "Choose columns to include in X (predictors):",
    available_features,
    default=available_features.tolist()  # Preselect all by default
)

# -------------------- Train-Test Split Ratio --------------------
st.subheader("📏 Select Train-Test Split Ratio")
test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, step=0.05)

# -------------------- Perform Split --------------------
if st.button("✂️ Perform Train-Test Split"):
    if not selected_features:
        st.error("❌ Please select at least one feature column.")
        st.stop()

    try:
        X = df[selected_features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Store everything in session state
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["target_column"] = target_col
        st.session_state["selected_features"] = selected_features

        st.success(f"✅ Train-Test split completed!")
        st.write("🔹 X_train shape:", X_train.shape)
        st.write("🔹 X_test shape:", X_test.shape)
        st.write("🔹 y_train shape:", y_train.shape)
        st.write("🔹 y_test shape:", y_test.shape)

    except Exception as e:
        st.error(f"❌ Error during splitting: {str(e)}")
