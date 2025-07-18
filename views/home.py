import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from utils.data_processing import apply_operations

def home():
    st.subheader("🏠 Home: Data Upload & Selection")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.markdown("**Or load a CSV from a GitHub link:**")
    github_link = st.text_input(
        "Paste the GitHub link to your CSV file",
        placeholder="e.g., https://raw.githubusercontent.com/marsgr6/r-scripts/master/data/viz_data/sample.csv"
    )
    load_from_github = st.button("Load from GitHub")

    st.markdown("""
    ⚠️ **Important:**  
    Your CSV file must be comma-separated, use a dot (`.`) as the decimal separator, and must not include thousands separators or special symbols.  
    Be aware that Excel often modifies CSV files when saving. Please ensure your file is as standard as possible before uploading.

    📎 **You can also try using sample CSV files:**  
    [Example Datasets on GitHub](https://github.com/marsgr6/r-scripts/tree/master/data/viz_data)
    """)

    if "operations" not in st.session_state:
        st.session_state["operations"] = []
    if "enable_operation_tracking" not in st.session_state:
        st.session_state["enable_operation_tracking"] = False
    if "current_numeric_cols" not in st.session_state:
        st.session_state["current_numeric_cols"] = []
    if "current_categorical_cols" not in st.session_state:
        st.session_state["current_categorical_cols"] = []
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False

    st.subheader("⚙️ Operation Tracking Settings")
    st.session_state["enable_operation_tracking"] = st.checkbox(
        "Enable Operation Tracking for Undo (May Slow Down for Large Datasets)",
        value=False,
        help="When enabled, operations are tracked, allowing you to undo them. This may slow down the app for large datasets as all operations are reapplied each time."
    )
    if st.session_state["enable_operation_tracking"]:
        st.warning("Operation tracking is enabled. This may impact performance for large datasets as all operations will be reapplied after each change.")

    if st.button("Reset All"):
        st.session_state["operations"] = []
        st.session_state.pop("raw_df", None)
        st.session_state.pop("filtered_df", None)
        st.session_state.pop("current_numeric_cols", None)
        st.session_state.pop("current_categorical_cols", None)
        st.session_state.pop("last_uploaded_file", None)
        st.session_state.pop("is_imputed", None)
        st.session_state.pop("imputed_df", None)
        st.session_state.pop("enable_operation_tracking", None)
        st.session_state["data_loaded"] = False
        st.success("Session reset successfully. Please upload a new file.")
        return

    if st.session_state["enable_operation_tracking"] and st.session_state["operations"]:
        if st.button("Undo Last Operation"):
            st.session_state["operations"].pop()
            if st.session_state["operations"] and "raw_df" in st.session_state:
                st.session_state["filtered_df"] = apply_operations(st.session_state["raw_df"], st.session_state["operations"])
                st.session_state["current_numeric_cols"] = st.session_state["filtered_df"].select_dtypes(include="number").columns.tolist()
                st.session_state["current_categorical_cols"] = st.session_state["filtered_df"].select_dtypes(include=["object", "category"]).columns.tolist()
                st.success("Last operation undone successfully!")
            else:
                st.session_state.pop("filtered_df", None)
                st.session_state.pop("current_numeric_cols", None)
                st.session_state.pop("current_categorical_cols", None)
                st.session_state["data_loaded"] = False
                st.info("No operations left to apply.")

    if load_from_github and github_link:
        try:
            if not github_link.endswith('.csv'):
                st.error("The provided link does not point to a CSV file. Please ensure the URL ends with '.csv'.")
            else:
                if "github.com" in github_link and "raw.githubusercontent.com" not in github_link:
                    github_link = github_link.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                response = requests.get(github_link)
                response.raise_for_status()
                csv_content = response.content.decode('utf-8')
                raw_df = pd.read_csv(io.StringIO(csv_content), na_values=['', 'NA', 'N/A', 'na', 'n/a', 'NaN', 'nan', 'NULL', 'null', 'missing', '-', '--'], keep_default_na=True)
                st.session_state["operations"] = []
                st.session_state["last_uploaded_file"] = github_link
                st.session_state.pop("filtered_df", None)
                st.session_state.pop("current_numeric_cols", None)
                st.session_state.pop("current_categorical_cols", None)
                st.session_state.pop("is_imputed", None)
                st.session_state.pop("imputed_df", None)
                st.session_state["raw_df"] = raw_df.copy()
                st.session_state["filtered_df"] = raw_df.copy()
                st.session_state["current_numeric_cols"] = raw_df.select_dtypes(include="number").columns.tolist()
                st.session_state["current_categorical_cols"] = raw_df.select_dtypes(include=["object", "category"]).columns.tolist()
                st.session_state["data_loaded"] = True
                st.success(f"CSV file successfully loaded from GitHub link: {github_link}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to load CSV from GitHub link: {e}")
        except pd.errors.EmptyDataError:
            st.error("The CSV file is empty or invalid.")
        except Exception as e:
            st.error(f"An error occurred while loading the CSV: {e}")

    if uploaded_file:
        na_values = ['', 'NA', 'N/A', 'na', 'n/a', 'NaN', 'nan', 'NULL', 'null', 'missing', '-', '--']
        if "last_uploaded_file" not in st.session_state or uploaded_file.name != st.session_state["last_uploaded_file"]:
            st.session_state["operations"] = []
            st.session_state["last_uploaded_file"] = uploaded_file.name
            st.session_state.pop("filtered_df", None)
            st.session_state.pop("current_numeric_cols", None)
            st.session_state.pop("current_categorical_cols", None)
            st.session_state.pop("is_imputed", None)
            st.session_state.pop("imputed_df", None)
        raw_df = pd.read_csv(uploaded_file, na_values=na_values, keep_default_na=True)
        st.success(f"File '{uploaded_file.name}' successfully uploaded!")
        st.session_state["raw_df"] = raw_df.copy()
        if "filtered_df" not in st.session_state:
            st.session_state["filtered_df"] = raw_df.copy()
            st.session_state["current_numeric_cols"] = raw_df.select_dtypes(include="number").columns.tolist()
            st.session_state["current_categorical_cols"] = raw_df.select_dtypes(include=["object", "category"]).columns.tolist()
        st.session_state["data_loaded"] = True

    if st.session_state["data_loaded"] and "filtered_df" in st.session_state:
        df = st.session_state["filtered_df"].copy()
        if st.session_state["enable_operation_tracking"] and "raw_df" in st.session_state and st.session_state["operations"]:
            df = apply_operations(st.session_state["raw_df"], st.session_state["operations"])
            st.session_state["filtered_df"] = df
            st.session_state["current_numeric_cols"] = df.select_dtypes(include="number").columns.tolist()
            st.session_state["current_categorical_cols"] = df.select_dtypes(include=["object", "category"]).columns.tolist()

        st.subheader("🎯 Column Selection")
        selected_columns = st.multiselect(
            "Select columns to keep",
            options=df.columns.tolist(),
            default=df.columns.tolist(),
            key="select_columns"
        )
        if st.button("Apply Column Selection"):
            if selected_columns:
                if st.session_state["enable_operation_tracking"]:
                    st.session_state["operations"].append({"type": "select_columns", "columns": selected_columns})
                    df = apply_operations(st.session_state["raw_df"], st.session_state["operations"])
                else:
                    df = df[selected_columns]
                st.session_state["filtered_df"] = df
                st.session_state["current_numeric_cols"] = df.select_dtypes(include="number").columns.tolist()
                st.session_state["current_categorical_cols"] = df.select_dtypes(include=["object", "category"]).columns.tolist()
                st.success("Column selection applied successfully!")
            else:
                st.warning("Please select at least one column.")

        st.subheader("🔧 Assign Variable Types")
        numeric_cols = st.multiselect(
            "Select Numeric Variables",
            options=df.columns.tolist(),
            default=st.session_state["current_numeric_cols"],
            key="numeric_cols"
        )
        categorical_cols = st.multiselect(
            "Select Categorical Variables",
            options=[col for col in df.columns if col not in numeric_cols],
            default=st.session_state["current_categorical_cols"],
            key="categorical_cols"
        )
        if st.button("Apply Variable Types"):
            if st.session_state["enable_operation_tracking"]:
                st.session_state["operations"].append({
                    "type": "cast_types",
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols
                })
                df = apply_operations(st.session_state["raw_df"], st.session_state["operations"])
            else:
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                for col in categorical_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str).replace('nan', np.nan)
            st.session_state["filtered_df"] = df
            st.session_state["current_numeric_cols"] = df.select_dtypes(include="number").columns.tolist()
            st.session_state["current_categorical_cols"] = df.select_dtypes(include=["object", "category"]).columns.tolist()
            st.success("Variable types applied successfully!")

        st.subheader("🧪 Random Subsampling")
        sample_pct = st.slider(
            "Select percentage of data to use",
            min_value=1,
            max_value=100,
            value=100,
            key="sample_pct"
        )
        if st.button("Apply Subsampling"):
            if st.session_state["enable_operation_tracking"]:
                st.session_state["operations"].append({"type": "subsample", "sample_pct": sample_pct})
                df = apply_operations(st.session_state["raw_df"], st.session_state["operations"])
            else:
                frac = sample_pct / 100
                df = df.sample(frac=frac, random_state=42)
            st.session_state["filtered_df"] = df
            st.session_state["current_numeric_cols"] = df.select_dtypes(include="number").columns.tolist()
            st.session_state["current_categorical_cols"] = df.select_dtypes(include=["object", "category"]).columns.tolist()
            st.success(f"Dataset subsampled to {len(df)} rows.")

        st.subheader("🔥 Optional: Melt Data")
        id_vars = st.multiselect(
            "Select ID Variables for Melting",
            options=df.columns.tolist(),
            key="melt_keys"
        )
        if st.button("Apply Melting"):
            if id_vars:
                if st.session_state["enable_operation_tracking"]:
                    st.session_state["operations"].append({"type": "melt", "id_vars": id_vars})
                    df = apply_operations(st.session_state["raw_df"], st.session_state["operations"])
                else:
                    value_vars = [col for col in df.columns if col not in id_vars]
                    df = pd.melt(
                        df,
                        id_vars=id_vars,
                        value_vars=value_vars,
                        var_name="variable",
                        value_name="value"
                    )
                st.session_state["filtered_df"] = df
                st.session_state["current_numeric_cols"] = df.select_dtypes(include="number").columns.tolist()
                st.session_state["current_categorical_cols"] = df.select_dtypes(include=["object", "category"]).columns.tolist()
                st.success("Melting applied successfully!")
            else:
                st.warning("Please select at least one ID variable before melting.")

        st.subheader("📅 Date Feature Engineering (Optional)")
        date_column = st.selectbox(
            "Select Date Column to Extract Features (optional)",
            options=["None"] + df.columns.tolist(),
            index=0,
            key="date_column"
        )
        dayfirst_option = st.checkbox("Use Day First when Parsing Dates (e.g., 31/12/2023)", value=True)
        if st.button("Parse Date and Create Features"):
            if date_column != "None":
                if st.session_state["enable_operation_tracking"]:
                    st.session_state["operations"].append({
                        "type": "date_features",
                        "date_column": date_column,
                        "dayfirst": dayfirst_option
                    })
                    df = apply_operations(st.session_state["raw_df"], st.session_state["operations"])
                else:
                    df[date_column] = pd.to_datetime(df[date_column], errors="coerce", dayfirst=dayfirst_option)
                    df["Year"] = df[date_column].dt.year
                    df["Month"] = df[date_column].dt.month
                    df["Day"] = df[date_column].dt.day
                    df["DayOfWeek"] = df[date_column].dt.dayofweek
                    df["Hour"] = df[date_column].dt.hour
                    df[date_column] = df[date_column].astype(str)
                st.session_state["filtered_df"] = df
                st.session_state["current_numeric_cols"] = df.select_dtypes(include="number").columns.tolist()
                st.session_state["current_categorical_cols"] = df.select_dtypes(include=["object", "category"]).columns.tolist()
                st.success("✅ Date features extracted successfully!")
            else:
                st.warning("Please select a date column to extract features.")

        if st.session_state["enable_operation_tracking"] and st.session_state["operations"]:
            st.subheader("📜 Operation History")
            for i, op in enumerate(st.session_state["operations"], 1):
                if op["type"] == "select_columns":
                    st.write(f"{i}. Selected columns: {', '.join(op['columns'])}")
                elif op["type"] == "cast_types":
                    st.write(f"{i}. Cast types - Numeric: {', '.join(op['numeric_cols'])}, Categorical: {', '.join(op['categorical_cols'])}")
                elif op["type"] == "subsample":
                    st.write(f"{i}. Subsampled to {op['sample_pct']}% of data")
                elif op["type"] == "melt":
                    st.write(f"{i}. Melted with ID vars: {', '.join(op['id_vars'])}")
                elif op["type"] == "date_features":
                    st.write(f"{i}. Extracted date features from {op['date_column']} (dayfirst: {op['dayfirst']})")

        st.subheader("📊 Preview of the Processed Dataset")
        if "filtered_df" in st.session_state:
            st.dataframe(st.session_state["filtered_df"].head())
        else:
            st.info("No processed data available. Please apply some operations.")

        st.subheader("💾 Download Processed Data")
        if "filtered_df" in st.session_state:
            csv = st.session_state["filtered_df"].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Processed Data as CSV",
                data=csv,
                file_name="processed_dataset.csv",
                mime="text/csv"
            )