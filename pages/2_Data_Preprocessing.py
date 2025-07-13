import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Step 2 - Data Preprocessing", layout="wide")
st.title("🧼 Step 2: Data Preprocessing")

# Load original data if not yet processed
if "raw_data" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first in Step 1.")
    st.stop()

# Initialize processed_data if not already done
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = st.session_state["raw_data"].copy()

df = st.session_state["processed_data"]

st.subheader("📊 Current Data Preview")
st.dataframe(df.head())

# -------------------- Missing Value Treatment --------------------
st.subheader("🚫 Missing Value Treatment")

missing_cols = df.columns[df.isnull().any()].tolist()

if missing_cols:
    selected_mv_cols = st.multiselect("👉 Select columns to treat missing values", missing_cols)

    if selected_mv_cols:
        mv_treatment_methods = {}
        for col in selected_mv_cols:
            mv_treatment_methods[col] = st.selectbox(
                f"Method for `{col}`:",
                ["Mean", "Median", "Mode", "Drop Column", "Drop Rows (with any missing)"],
                key=f"mv_{col}"
            )

        if st.button("✅ Apply Missing Value Treatment"):
            updated_df = df.copy()
            for col, method in mv_treatment_methods.items():
                if method == "Mean":
                    updated_df[col].fillna(updated_df[col].mean(), inplace=True)
                elif method == "Median":
                    updated_df[col].fillna(updated_df[col].median(), inplace=True)
                elif method == "Mode":
                    updated_df[col].fillna(updated_df[col].mode()[0], inplace=True)
                elif method == "Drop Column":
                    updated_df.drop(columns=[col], inplace=True)
                elif method == "Drop Rows (with any missing)":
                    updated_df.dropna(inplace=True)
                    break
            st.session_state["processed_data"] = updated_df
            st.success("✅ Missing value treatment applied.")
            st.dataframe(updated_df.head())
else:
    st.info("✅ No missing values found.")

# -------------------- Outlier Treatment --------------------
st.subheader("📈 Outlier Treatment (IQR method)")

numeric_cols = df.select_dtypes(include='number').columns.tolist()

if numeric_cols:
    if st.button("📉 Remove Outliers (IQR Method)"):
        updated_df = df.copy()
        initial_shape = updated_df.shape
        for col in numeric_cols:
            Q1 = updated_df[col].quantile(0.25)
            Q3 = updated_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            updated_df = updated_df[(updated_df[col] >= lower) & (updated_df[col] <= upper)]
        st.session_state["processed_data"] = updated_df
        st.success(f"✅ Outliers removed. Rows reduced from {initial_shape[0]} to {updated_df.shape[0]}.")
        st.dataframe(updated_df.head())
else:
    st.info("ℹ️ No numeric columns available for outlier detection.")

# -------------------- Encoding Categorical Columns --------------------
st.subheader("🔠 Encoding Categorical Columns")

cat_cols = df.select_dtypes(include='object').columns.tolist()

if cat_cols:
    selected_cat_cols = st.multiselect("👉 Select columns to encode", cat_cols)

    if selected_cat_cols:
        encoding_choices = {}
        for col in selected_cat_cols:
            encoding_choices[col] = st.selectbox(
                f"Encoding method for `{col}`:",
                ["Label Encoding", "One-Hot Encoding"],
                key=f"encode_{col}"
            )

        if st.button("🧪 Apply Encoding"):
            updated_df = df.copy()
            for col, method in encoding_choices.items():
                if method == "Label Encoding":
                    le = LabelEncoder()
                    updated_df[col] = le.fit_transform(updated_df[col])
                elif method == "One-Hot Encoding":
                    updated_df = pd.get_dummies(updated_df, columns=[col], dtype=int, drop_first=True)
            st.session_state["processed_data"] = updated_df
            st.success("✅ Encoding applied.")
            st.dataframe(updated_df.head())
else:
    st.info("✅ No categorical columns found.")

# -------------------- Final Output --------------------
st.subheader("✅ Final Processed Data Stored")
st.dataframe(st.session_state["processed_data"].head())
