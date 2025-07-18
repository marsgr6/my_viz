import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processing import apply_operations

def missing_data_imputation():
    st.subheader("🔄 Missing Data Imputation")

    if "filtered_df" not in st.session_state:
        st.warning("Please upload and process a dataset in the Home section first.")
        return

    df = st.session_state["filtered_df"].copy()
    df = df.apply(lambda x: x.replace([None], np.nan))

    st.subheader("📉 Missing Data Overview")
    missing_data = df.isna().sum()
    missing_percentage = (missing_data / len(df) * 100).round(2)
    missing_info = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_data,
        "Percentage Missing (%)": missing_percentage
    })
    st.dataframe(missing_info)

    if missing_data.sum() == 0:
        st.info("No missing values found in the dataset.")
        st.session_state["is_imputed"] = False
        return

    st.subheader("🧹 Data Cleaning Options")
    if st.button("Remove Rows with Any Missing Values"):
        original_rows = df.shape[0]
        if st.session_state["enable_operation_tracking"]:
            st.session_state["operations"].append({"type": "remove_nan"})
            df = apply_operations(st.session_state["raw_df"], st.session_state["operations"])
        else:
            df = df.dropna()
        st.session_state["filtered_df"] = df
        st.session_state["current_numeric_cols"] = st.session_state["filtered_df"].select_dtypes(include="number").columns.tolist()
        st.session_state["current_categorical_cols"] = st.session_state["filtered_df"].select_dtypes(include=["object", "category"]).columns.tolist()
        st.success(f"Removed rows with missing values! {original_rows - df.shape[0]} rows dropped. New size: {df.shape[0]} rows.")
        missing_data = df.isna().sum()
        missing_percentage = (missing_data / len(df) * 100).round(2)
        missing_info = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": missing_data,
            "Percentage Missing (%)": missing_percentage
        })
        st.dataframe(missing_info)
        if missing_data.sum() == 0:
            st.info("No missing values left after removal.")
            st.session_state["is_imputed"] = False
            return

    st.subheader("🔧 Imputation Settings")
    if st.button("Impute Missing Values with KNN Imputer (Numeric) and Mode (Categorical)"):
        if st.session_state["enable_operation_tracking"]:
            st.session_state["operations"].append({"type": "impute"})
            df = apply_operations(st.session_state["raw_df"], st.session_state["operations"])
        else:
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            numeric_df = df[numeric_cols].copy()
            categorical_df = df[categorical_cols].copy()
            if numeric_cols:
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")
                imputed_numeric = imputer.fit_transform(numeric_df)
                imputed_numeric_df = pd.DataFrame(imputed_numeric, columns=numeric_cols, index=numeric_df.index)
            else:
                imputed_numeric_df = pd.DataFrame(index=df.index)
            if categorical_cols:
                for col in categorical_cols:
                    most_frequent = categorical_df[col].mode()
                    if not most_frequent.empty:
                        categorical_df[col].fillna(most_frequent[0], inplace=True)
            imputed_categorical_df = categorical_df
            df = pd.concat([imputed_numeric_df, imputed_categorical_df], axis=1)
            df = df[df.columns]
        st.session_state["filtered_df"] = df
        st.session_state["current_numeric_cols"] = st.session_state["filtered_df"].select_dtypes(include="number").columns.tolist()
        st.session_state["current_categorical_cols"] = st.session_state["filtered_df"].select_dtypes(include=["object", "category"]).columns.tolist()
        st.success("Missing values have been imputed!")
        st.write("📊 Preview of the imputed dataset:")
        st.dataframe(df.head())
        st.session_state["is_imputed"] = True

    if "is_imputed" in st.session_state and st.session_state["is_imputed"]:
        st.info("Currently using imputed dataset for visualization.")
    else:
        st.info("Currently using original dataset for visualization. Click the button above to impute missing values.")