import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def apply_operations(raw_df, operations):
    df = raw_df.copy()
    for op in operations:
        try:
            if op["type"] == "select_columns":
                df = df[op["columns"]]
            elif op["type"] == "cast_types":
                for col in op["numeric_cols"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                for col in op["categorical_cols"]:
                    if col in df.columns:
                        df[col] = df[col].astype(str).replace('nan', np.nan)
            elif op["type"] == "subsample":
                frac = op["sample_pct"] / 100
                df = df.sample(frac=frac, random_state=42)
            elif op["type"] == "melt":
                value_vars = [col for col in df.columns if col not in op["id_vars"]]
                df = pd.melt(
                    df,
                    id_vars=op["id_vars"],
                    value_vars=value_vars,
                    var_name="variable",
                    value_name="value"
                )
            elif op["type"] == "date_features":
                df[op["date_column"]] = pd.to_datetime(df[op["date_column"]], errors="coerce", dayfirst=op["dayfirst"])
                df["Year"] = df[op["date_column"]].dt.year
                df["Month"] = df[op["date_column"]].dt.month
                df["Day"] = df[op["date_column"]].dt.day
                df["DayOfWeek"] = df[op["date_column"]].dt.dayofweek
                df["Hour"] = df[op["date_column"]].dt.hour
                df[op["date_column"]] = df[op["date_column"]].astype(str)
            elif op["type"] == "remove_nan":
                df = df.dropna()
            elif op["type"] == "impute":
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
                categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                numeric_df = df[numeric_cols].copy()
                categorical_df = df[categorical_cols].copy()
                if numeric_cols:
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
        except Exception as e:
            return df
    return df