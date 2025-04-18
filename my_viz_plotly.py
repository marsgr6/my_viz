import streamlit as st
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.cluster import hierarchy
from sklearn.impute import KNNImputer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

# --- HOME SECTION ---
def home():
    st.subheader("🏠 Home: Data Upload & Selection")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Define common missing value representations
        na_values = ['', 'NA', 'N/A', 'na', 'n/a', 'NaN', 'nan', 'NULL', 'null', 'missing', '-', '--']

        # Read CSV with specified missing values
        df = pd.read_csv(uploaded_file, na_values=na_values, keep_default_na=True)
        st.success("File successfully uploaded!")

        # Display column info
        st.subheader("🔍 Column Overview")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": [df[col].dtype for col in df.columns],
            "Unique Values": [df[col].nunique(dropna=True) for col in df.columns],
            "Missing Values": [df[col].isna().sum() for col in df.columns]
        })
        st.dataframe(col_info)

        # Column selection
        selected_columns = st.multiselect(
            "Select columns to keep", options=df.columns, default=df.columns.tolist()
        )
        df = df[selected_columns]

        # Manual type assignment
        st.subheader("🔧 Assign Variable Types")
        numeric_cols = st.multiselect("Select numeric variables", options=selected_columns,
                                      default=df.select_dtypes(include='number').columns.tolist())
        categorical_cols = st.multiselect("Select categorical variables", options=[col for col in selected_columns if col not in numeric_cols],
                                          default=df.select_dtypes(include='object').columns.tolist())

        # Convert columns to their assigned types
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce invalids to NaN

        for col in categorical_cols:
            df[col] = df[col].astype(str).replace('nan', np.nan)  # Convert to string, preserve NaN

        # Subsampling
        st.subheader("🧪 Random Subsampling")
        sample_pct = st.slider("Select percentage of data to use", min_value=1, max_value=100, value=100)
        if sample_pct < 100:
            df = df.sample(frac=sample_pct / 100, random_state=42)
            st.info(f"Dataset has been subsampled to {len(df)} rows.")

        st.write("📊 Preview of the filtered dataset:")
        st.dataframe(df.head())

        # Save filtered info in session state
        st.session_state["filtered_df"] = df
        st.session_state["numeric_cols"] = numeric_cols
        st.session_state["categorical_cols"] = categorical_cols
        # Reset imputation state when new data is uploaded
        if "is_imputed" in st.session_state:
            del st.session_state["is_imputed"]
        if "imputed_df" in st.session_state:
            del st.session_state["imputed_df"]
        if "working_df" in st.session_state:
            del st.session_state["working_df"]

# --- MISSING DATA IMPUTATION SECTION ---
def missing_data_imputation():
    st.subheader("🔄 Missing Data Imputation")

    # Check if filtered data exists
    if "filtered_df" not in st.session_state:
        st.warning("Please upload and process a dataset in the Home section first.")
        return

    # Retrieve filtered data
    if "working_df" not in st.session_state:
        st.session_state["working_df"] = st.session_state["filtered_df"].copy()

    df = st.session_state["working_df"]

    # Standardize None values to np.nan
    df = df.applymap(lambda x: np.nan if x is None else x)

    # Update working copy after standardization
    st.session_state["working_df"] = df

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Calculate missing data statistics
    st.subheader("📉 Missing Data Overview")
    missing_data = df.isna().sum()
    missing_percentage = (missing_data / len(df) * 100).round(2)
    missing_info = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_data,
        "Percentage Missing (%)": missing_percentage
    })
    st.dataframe(missing_info)

    # Check if there are any missing values
    if missing_data.sum() == 0:
        st.info("No missing values found in the dataset.")
        st.session_state["is_imputed"] = False
        return

    # Remove rows with NaN (button)
    st.subheader("🧹 Data Cleaning Options")
    remove_nan_button = st.button("Remove Rows with Any Missing Values")

    if remove_nan_button:
        original_rows = df.shape[0]
        df = df.dropna()
        st.session_state["working_df"] = df  # Update working data
        st.success(f"Removed rows with missing values! {original_rows - df.shape[0]} rows dropped. New size: {df.shape[0]} rows.")

        # Recalculate missing info after removal
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

    # KNN Imputation options
    st.subheader("🔧 Imputation Settings")
    impute_button = st.button("Impute Missing Values with KNN Imputer (Numeric) and Mode (Categorical)")

    if impute_button:
        # Separate numeric and categorical data again
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        numeric_df = df[numeric_cols].copy()
        categorical_df = df[categorical_cols].copy()

        # 1. Numeric Columns: KNN Imputation
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")
            imputed_numeric = imputer.fit_transform(numeric_df)
            imputed_numeric_df = pd.DataFrame(imputed_numeric, columns=numeric_cols, index=numeric_df.index)
        else:
            imputed_numeric_df = pd.DataFrame(index=df.index)

        # 2. Categorical Columns: Mode Imputation
        if categorical_cols:
            for col in categorical_cols:
                most_frequent = categorical_df[col].mode()
                if not most_frequent.empty:
                    categorical_df[col].fillna(most_frequent[0], inplace=True)
        imputed_categorical_df = categorical_df

        # Combine imputed numeric data with categorical data
        imputed_df = pd.concat([imputed_numeric_df, imputed_categorical_df], axis=1)

        # Ensure column order matches original dataframe
        imputed_df = imputed_df[df.columns]

        # Store the imputed dataframe in session state
        st.session_state["imputed_df"] = imputed_df
        st.session_state["is_imputed"] = True

        st.success("Missing values have been imputed!")
        st.write("📊 Preview of the imputed dataset:")
        st.dataframe(imputed_df.head())

    # Display current imputation status
    if "is_imputed" in st.session_state and st.session_state["is_imputed"]:
        st.info("Currently using imputed dataset for visualization.")
    else:
        st.info("Currently using original dataset for visualization. Click the button above to impute missing values.")

# --- VISUALIZATION SECTION ---
def visualization():
    st.subheader("🧪 Visualization: Interactive Data Exploration")

    # Check if filtered data exists in session state
    if "filtered_df" not in st.session_state:
        st.warning("Please upload and process a dataset in the Home section first.")
        return

    # Decide which dataset to use based on imputation status
    if "is_imputed" in st.session_state and st.session_state["is_imputed"]:
        data = st.session_state["imputed_df"]
        st.success("Using imputed dataset for visualization.")
    elif "working_df" in st.session_state:
        data = st.session_state["working_df"]
        st.info("Using cleaned dataset (after removing NaNs) for visualization.")
    else:
        data = st.session_state["filtered_df"]
        st.info("Using original uploaded dataset for visualization.")


    numeric_cols = st.session_state["numeric_cols"]
    categorical_cols = st.session_state["categorical_cols"]

    # Convert categorical columns to strings
    for col in data.select_dtypes(include='category').columns:
        data[col] = data[col].astype('str')

    # Define plot types
    PLOT_TYPES = [
        'bars', 'boxes', 'ridges', 'histogram', 'density 1', 'density 2', 
        'scatter', 'catplot', 'missingno', 'correlation', 'clustermap', 
        'pairplot', 'regression'
    ]

    if not numeric_cols:
        # Define plot types
        PLOT_TYPES = [
            'bars', 'boxes', 'histogram', 'density 2', 
            'scatter', 'catplot', 'missingno'
        ]


    # Streamlit UI elements for plot selection (in sidebar)
    st.sidebar.header("Plot Selection")
    plot_type = st.sidebar.selectbox("Select Plot Type", PLOT_TYPES)
    risk_it_all = st.sidebar.checkbox("Risk It All", value=False)

    # Define color palette (Plotly equivalent of Set2)
    PALETTE = px.colors.qualitative.Set2

    # Plot rendering function
    def render_plot():
        # Non-interactive plot types
        if plot_type == "correlation":
            import numpy as np
            import plotly.graph_objects as go
            data_c = data.drop(data.columns[data.nunique() == 1], axis=1)
            corr_matrix = data_c.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis', 
                           title="Correlation Matrix", width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "clustermap":
            import numpy as np
            import plotly.graph_objects as go
            z_score = st.sidebar.selectbox("Z-Score", [None, 0, 1], index=0)
            standard_scale = st.sidebar.selectbox("Standard Scale", [None, 0, 1], index=0)
            data_c = data.drop(data.columns[data.nunique() == 1], axis=1)
            numeric_data = data_c.select_dtypes(include='number').dropna()

            if z_score is not None:
                numeric_data = (numeric_data - numeric_data.mean()) / numeric_data.std()
            if standard_scale is not None:
                numeric_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
            
            # Compute correlation matrix
            corr_matrix = numeric_data.corr()
            labels = corr_matrix.columns
            
            # Compute dendrograms
            row_linkage = hierarchy.linkage(corr_matrix, method='average')
            col_linkage = hierarchy.linkage(corr_matrix.T, method='average')
            row_order = hierarchy.leaves_list(row_linkage)
            col_order = hierarchy.leaves_list(col_linkage)
            corr_matrix = corr_matrix.iloc[row_order, col_order]
            labels = corr_matrix.columns
            
            # Create figure with heatmap
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=labels,
                    y=labels,
                    colorscale='Viridis',
                    showscale=True,
                    text=corr_matrix.values.round(2),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                )
            )
            fig.update_layout(
                title="Clustermap with Clustering",
                width=800,
                height=800,
                xaxis=dict(side="top", showticklabels=True),
                yaxis=dict(showticklabels=True),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # pairplot
        elif plot_type == "pairplot":
            import numpy as np
            import plotly.graph_objects as go

            numeric_cols_list = list(data.select_dtypes(include='number').columns)
            cat_cols = list(data.select_dtypes(include=['object', 'category', 'bool']).columns)

            if not numeric_cols_list:
                st.warning("No numeric columns available for pairplot.")
                return

            # Hue selection only if there are categorical columns
            has_categorical = len(cat_cols) > 0

            if has_categorical:
                hue_cols = [None] + (data.columns.tolist() if risk_it_all else cat_cols)
                hue_var = st.sidebar.selectbox(
                    "Hue Variable", hue_cols, index=0,
                    format_func=lambda x: "None" if x is None else x
                )
            else:
                hue_var = None

            # Create subplot grid without shared axes
            n_cols = len(numeric_cols_list)
            fig = make_subplots(
                rows=n_cols, cols=n_cols,
                row_titles=numeric_cols_list,
                column_titles=numeric_cols_list,
                vertical_spacing=0.05,
                horizontal_spacing=0.05,
                shared_xaxes=False,
                shared_yaxes=False
            )

            # Plot KDEs on diagonal and scatters off-diagonal
            for i in range(n_cols):
                col_i = numeric_cols_list[i]
                for j in range(n_cols):
                    col_j = numeric_cols_list[j]

                    # Diagonal: KDEs by hue
                    if i == j:
                        if hue_var is None:
                            # Single KDE without hue
                            subset = data[col_i].dropna()
                            if len(subset) >= 2:
                                kde = gaussian_kde(subset)
                                x_values = np.linspace(min(subset), max(subset), 100)
                                density = kde(x_values)
                                density = density / density.max()
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_values,
                                        y=density,
                                        mode='lines',
                                        name=col_i,
                                        line=dict(color=PALETTE[0]),
                                        showlegend=(i == 0 and j == 0)
                                    ),
                                    row=i+1, col=j+1
                                )
                        else:
                            for k, hue_val in enumerate(data[hue_var].dropna().unique()):
                                subset = data[data[hue_var] == hue_val][col_i].dropna()
                                if len(subset) < 2:
                                    continue
                                try:
                                    kde = gaussian_kde(subset)
                                    x_values = np.linspace(min(subset), max(subset), 100)
                                    density = kde(x_values)
                                    density = density / density.max()
                                    fig.add_trace(
                                        go.Scatter(
                                            x=x_values,
                                            y=density,
                                            mode='lines',
                                            name=str(hue_val),
                                            line=dict(color=PALETTE[k % len(PALETTE)]),
                                            showlegend=(i == 0 and j == 0)
                                        ),
                                        row=i+1, col=j+1
                                    )
                                except Exception:
                                    continue

                    # Off-diagonal: Scatter plots
                    else:
                        if hue_var is None:
                            subset = data[[col_j, col_i]].dropna()
                            if subset.empty:
                                continue
                            fig.add_trace(
                                go.Scatter(
                                    x=subset[col_j],
                                    y=subset[col_i],
                                    mode='markers',
                                    marker=dict(color=PALETTE[0], size=5),
                                    name="Data",
                                    showlegend=False
                                ),
                                row=i+1, col=j+1
                            )
                        else:
                            for k, hue_val in enumerate(data[hue_var].dropna().unique()):
                                subset = data[data[hue_var] == hue_val][[col_j, col_i]].dropna()
                                if subset.empty:
                                    continue
                                fig.add_trace(
                                    go.Scatter(
                                        x=subset[col_j],
                                        y=subset[col_i],
                                        mode='markers',
                                        marker=dict(color=PALETTE[k % len(PALETTE)], size=5),
                                        name=str(hue_val),
                                        showlegend=False
                                    ),
                                    row=i+1, col=j+1
                                )

            # Update layout
            fig.update_layout(
                title="Pairplot (KDE Diagonals, Scatter Off-Diagonals)",
                width=800,
                height=800,
                showlegend=True
            )

            # Update axes ranges for each subplot independently
            for i in range(n_cols):
                for j in range(n_cols):
                    col_i = numeric_cols_list[i]
                    col_j = numeric_cols_list[j]
                    x_data = data[col_j].dropna()
                    y_data = data[col_i].dropna()
                    x_range = [min(x_data), max(x_data)] if not x_data.empty else [0, 1]
                    y_range = [0, 1.2] if i == j else [min(y_data), max(y_data)] if not y_data.empty else [0, 1]
                    x_margin = (x_range[1] - x_range[0]) * 0.05
                    y_margin = (y_range[1] - y_range[0]) * 0.05 if i != j else 0.1
                    x_range = [x_range[0] - x_margin, x_range[1] + x_margin]
                    y_range = [y_range[0] - y_margin, y_range[1] + y_margin]
                    fig.update_xaxes(range=x_range, row=i+1, col=j+1)
                    fig.update_yaxes(range=y_range, row=i+1, col=j+1)

            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "missingno":
            tplot = st.sidebar.selectbox("Plot Type", ["matrix", "bars", "heatmap", "dendrogram"])
            plt.close('all')  # Close any existing figures
            fig, ax = plt.subplots(figsize=(12, 8))  # Explicitly create figure and axes
            try:
                if tplot == "matrix":
                    msno.matrix(data, ax=ax)
                elif tplot == "bars":
                    msno.bar(data, ax=ax)
                elif tplot == "heatmap":
                    msno.heatmap(data, ax=ax)
                elif tplot == "dendrogram":
                    msno.dendrogram(data, ax=ax)
                plt.tight_layout()  # Adjust layout to prevent cutoff
                st.pyplot(fig)  # Render Matplotlib figure for missingno
            except Exception as e:
                st.error(f"Error rendering {tplot} plot: {e}")
            finally:
                plt.close(fig)  # Clean up figure

        # Interactive plot types with risk_it_all toggle
        else:
            # Bars
            if plot_type == 'bars':
                if categorical_cols:
                    x_cols =  data.columns.tolist() if risk_it_all else categorical_cols
                else:
                    x_cols = numeric_cols

                # Safe column lists
                hue_cols = [None] + (data.columns.tolist() if risk_it_all else categorical_cols)

                var_x = st.sidebar.selectbox(
                    "X Variable", x_cols, index=0,
                    format_func=lambda x: "None" if x is None else x
                )
                hue = st.sidebar.selectbox(
                    "Hue (Color)", hue_cols, index=0,
                    format_func=lambda x: "None" if x is None else x
                )

                tplot = st.sidebar.selectbox("Plot Type", ["bars", "heatmap"])
                bar_mode = st.sidebar.selectbox("Bars Mode", ["Histogram", "Raw Values (height = value in order)"])

                # --- NEW --- Time is here!
                time_mode = st.sidebar.checkbox("Time is here! (Index X, Select Y)", value=False)
                if time_mode:
                    y_cols = data.columns.tolist()
                    var_y = st.sidebar.selectbox("Y Variable (for Raw Values)", y_cols, index=0)

                plot_data = data.copy()

                # Force Hue column to string if selected
                if hue is not None:
                    plot_data[hue] = plot_data[hue].astype(str)

                if tplot == "bars":
                    if bar_mode == "Histogram" and not time_mode:
                        if hue is not None:
                            fig = px.histogram(
                                plot_data,
                                x=var_x,
                                color=hue,
                                barmode='group',
                                color_discrete_sequence=PALETTE,
                                width=800,
                                height=600
                            )
                        else:
                            fig = px.histogram(
                                plot_data,
                                x=var_x,
                                color_discrete_sequence=PALETTE,
                                width=800,
                                height=600
                            )
                        fig.update_traces(texttemplate='%{y}', textposition='auto')

                    else:  # Raw Values mode or Time Mode
                        df_plot = plot_data.reset_index()

                        if time_mode:
                            x_axis = var_x
                            y_axis = var_y
                        else:
                            x_axis = 'index'
                            y_axis = var_x

                        if y_axis is None:
                            st.warning("Please select a variable for Y-axis.")
                            return

                        if hue is not None:
                            fig = px.bar(
                                df_plot,
                                x=x_axis,
                                y=y_axis,
                                color=hue,
                                color_discrete_sequence=PALETTE,
                                width=800,
                                height=600
                            )
                        else:
                            fig = px.bar(
                                df_plot,
                                x=x_axis,
                                y=y_axis,
                                color_discrete_sequence=PALETTE,
                                width=800,
                                height=600
                            )
                        fig.update_layout(xaxis_title="Index", yaxis_title=y_axis)

                else:  # heatmap
                    if hue is not None and var_x is not None:
                        if pd.api.types.is_numeric_dtype(plot_data[var_x]):
                            plot_data[var_x] = pd.cut(plot_data[var_x], bins=10)
                        if pd.api.types.is_numeric_dtype(plot_data[hue]):
                            plot_data[hue] = pd.cut(plot_data[hue], bins=10)

                        df_2dhist = pd.crosstab(plot_data[var_x], plot_data[hue])

                        fig = px.imshow(
                            df_2dhist,
                            text_auto='.0f',
                            color_continuous_scale='Viridis',
                            title=f"Heatmap: {hue} vs {var_x}",
                            width=800,
                            height=600
                        )
                        fig.update_xaxes(title=hue)
                        fig.update_yaxes(title=var_x)
                    else:
                        st.warning("You must select both X and Hue variables to create a heatmap.")
                        return

                st.plotly_chart(fig, use_container_width=True)

            # Boxes
            if plot_type == 'boxes':
                import numpy as np  # Safe import for swarm

                x_cols = data.columns if risk_it_all else categorical_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = ['None'] + (data.columns if risk_it_all else categorical_cols)
                facet_cols = ['None'] + (categorical_cols if not risk_it_all else data.columns.tolist())

                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0)
                facet_col = st.sidebar.selectbox("Facet Column", facet_cols, index=0)
                facet_row = st.sidebar.selectbox("Facet Row", facet_cols, index=0)
                tplot = st.sidebar.selectbox("Plot Type", ["boxplot", "lineplot", "violin"])

                swarm_points = st.sidebar.checkbox("Overlay Swarm of Points", value=False)

                plot_data = data.copy()

                # Base plot arguments
                plot_kwargs = dict(
                    data_frame=plot_data,
                    y=var_y,
                    color_discrete_sequence=PALETTE,
                    width=800,
                    height=600,
                )

                if var_x != 'None':
                    plot_kwargs['x'] = var_x
                if hue != 'None':
                    plot_kwargs['color'] = hue
                if facet_col != 'None':
                    plot_kwargs['facet_col'] = facet_col
                if facet_row != 'None':
                    plot_kwargs['facet_row'] = facet_row

                if tplot == "boxplot":
                    fig = px.box(**plot_kwargs)
                elif tplot == "violin":
                    fig = px.violin(**plot_kwargs, box=True)
                elif tplot == "lineplot":
                    if hue != 'None':
                        group_cols = [var_x, hue] if var_x != 'None' else [hue]
                    else:
                        group_cols = [var_x] if var_x != 'None' else []

                    line_data = plot_data.groupby(group_cols)[var_y].mean().reset_index()

                    line_kwargs = dict(
                        data_frame=line_data,
                        y=var_y,
                        color_discrete_sequence=PALETTE,
                        width=800,
                        height=600,
                    )
                    if var_x != 'None':
                        line_kwargs['x'] = var_x
                    if hue != 'None':
                        line_kwargs['color'] = hue
                    if facet_col != 'None':
                        line_kwargs['facet_col'] = facet_col
                    if facet_row != 'None':
                        line_kwargs['facet_row'] = facet_row

                    fig = px.line(**line_kwargs)

                # Optionally add swarm points
                if swarm_points and tplot in ["boxplot", "violin"]:
                    import plotly.graph_objects as go

                    unique_hues = plot_data[hue].unique() if hue != 'None' else [None]

                    for i, hue_val in enumerate(unique_hues):
                        if hue_val is not None:
                            subset = plot_data[plot_data[hue] == hue_val]
                        else:
                            subset = plot_data

                        if var_x != 'None':
                            x_values = subset[var_x]
                        else:
                            x_values = np.zeros(len(subset))  # Center at 0 if no x

                        # Add scatter points as swarm
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=subset[var_y],
                                mode='markers',
                                marker=dict(
                                    color=PALETTE[i % len(PALETTE)],
                                    size=5,
                                    opacity=0.6,
                                    line=dict(width=0)
                                ),
                                name=str(hue_val) if hue_val is not None else "",
                                showlegend=False
                            )
                        )

                st.plotly_chart(fig, use_container_width=True)

            # Ridges
            elif plot_type == 'ridges':
                import numpy as np
                import plotly.graph_objects as go

                x_cols = data.columns if risk_it_all else categorical_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                cat_cols = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

                only_numeric = len(cat_cols) == 0  # <- check if only numeric

                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)

                if not only_numeric:
                    var_x = st.sidebar.selectbox("X Variable (for Ridges)", x_cols, index=0)
                    hue_cols = data.columns if risk_it_all else categorical_cols
                    hue_var = st.sidebar.selectbox("Hue Variable", hue_cols, index=0)
                    no_hue = st.sidebar.checkbox("No Hue", value=False)
                    hue_param = var_x if no_hue else hue_var

                    # Get unique var_x values for faceting
                    x_values_unique = data[var_x].dropna().unique()
                    n_rows = len(x_values_unique)
                    if n_rows == 0:
                        st.warning("No valid categories in X variable.")
                        return

                    # Create figure with subplots (one row per var_x value)
                    fig = make_subplots(
                        rows=n_rows, cols=1,
                        row_titles=[str(x) for x in x_values_unique],
                        shared_xaxes=True,
                        vertical_spacing=0.05
                    )

                    x_range = [data[var_y].min(), data[var_y].max()]
                    x_values = np.linspace(x_range[0], x_range[1], 100)

                    # Plot KDE for each var_x and hue combination
                    for row_idx, x_val in enumerate(x_values_unique, 1):
                        subset = data[data[var_x] == x_val]
                        hue_values = subset[hue_param].dropna().unique() if hue_param != var_x else [x_val]
                        for j, hue_val in enumerate(hue_values):
                            hue_subset = subset[subset[hue_param] == hue_val] if hue_param != var_x else subset
                            y_data = hue_subset[var_y].dropna()
                            if len(y_data) < 2:
                                continue
                            try:
                                kde = gaussian_kde(y_data)
                                density = kde(x_values)
                                density = density / density.max() * 0.4  # Normalize height
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_values,
                                        y=density,
                                        mode='lines',
                                        fill='tozeroy',
                                        name=f"{hue_val}" if hue_param != var_x else str(x_val),
                                        line=dict(color=PALETTE[j % len(PALETTE)]),
                                        showlegend=(row_idx == 1)
                                    ),
                                    row=row_idx, col=1
                                )
                            except Exception:
                                continue

                    fig.update_layout(
                        title=f"Ridge Plot Faceted by {var_x}",
                        width=800,
                        height=150 * n_rows,
                        showlegend=True
                    )
                    fig.update_xaxes(title=var_y, row=n_rows, col=1)
                    for row in range(1, n_rows + 1):
                        fig.update_yaxes(title="Density", range=[0, 0.5], row=row, col=1, showticklabels=False)

                else:
                    # Only numeric -> simple KDE plot
                    y_data = data[var_y].dropna()
                    if len(y_data) < 2:
                        st.warning("Not enough data points for density plot.")
                        return

                    kde = gaussian_kde(y_data)
                    x_values = np.linspace(y_data.min(), y_data.max(), 200)
                    density = kde(x_values)

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=density / density.max(),  # Normalize
                            mode='lines',
                            fill='tozeroy',
                            name=var_y,
                            line=dict(color=PALETTE[0])
                        )
                    )

                    fig.update_layout(
                        title=f"Univariate Density: {var_y}",
                        width=800,
                        height=600,
                        showlegend=False,
                        xaxis_title=var_y,
                        yaxis_title="Density"
                    )

                st.plotly_chart(fig, use_container_width=True)

            # Histogram
            elif plot_type == 'histogram':
                x_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = data.columns if risk_it_all else categorical_cols
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                hue_var = st.sidebar.selectbox("Hue Variable", hue_cols, index=0)
                multiple = st.sidebar.selectbox("Multiple", ["layer", "dodge", "stack"])
                stat = st.sidebar.selectbox("Stat", ["count", "probability", "percent", "density"])
                element = st.sidebar.selectbox("Element", ["bars", "step"])
                common_norm = st.sidebar.checkbox("Common Norm", value=False)
                cumulative = st.sidebar.checkbox("Cumulative", value=False)
                barmode = 'overlay' if multiple == 'layer' else multiple
                if element == "step":
                    fig = go.Figure()
                    for hue_val in data[hue_var].unique() if hue_var else [None]:
                        subset = data[data[hue_var] == hue_val] if hue_var else data
                        fig.add_trace(
                            go.Histogram(
                                x=subset[var_x],
                                histnorm=stat if stat != "count" else None,
                                cumulative_enabled=cumulative,
                                opacity=0.75 if multiple == 'layer' else 1.0,
                                name=str(hue_val) if hue_var else var_x,
                                histfunc="sum",
                                bingroup=1 if common_norm else None
                            )
                        )
                    fig.update_layout(
                        barmode=barmode,
                        width=800,
                        height=600,
                        title="Step Histogram",
                        xaxis_title=var_x,
                        yaxis_title=stat.capitalize(),
                        showlegend=True
                    )
                else:
                    fig = px.histogram(data, x=var_x, color=hue_var, barmode=barmode, 
                                      histnorm=stat if stat != "count" else None, 
                                      cumulative=cumulative, 
                                      opacity=0.75 if multiple == 'layer' else 1.0, 
                                      color_discrete_sequence=PALETTE, 
                                      width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)

            # Density 1 (Safe Univariate or Hue Density)
            elif plot_type == 'density 1':
                import numpy as np
                import plotly.graph_objects as go  # ← THIS was missing!

                x_cols = data.columns if risk_it_all else numeric_cols
                cat_cols = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                has_categorical = len(cat_cols) > 0

                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)

                if has_categorical:
                    hue_cols = [None] + (data.columns.tolist() if risk_it_all else cat_cols)
                    hue_var = st.sidebar.selectbox(
                        "Hue Variable", hue_cols, index=0,
                        format_func=lambda x: "None" if x is None else x
                    )
                else:
                    hue_var = None  # No hue option shown

                multiple = st.sidebar.selectbox("Multiple", ["layer", "stack"])
                common_norm = st.sidebar.checkbox("Common Norm (Normalize)", value=False)
                cumulative = st.sidebar.checkbox("Cumulative Density", value=False)

                fig = go.Figure()
                x_values = np.linspace(data[var_x].min(), data[var_x].max(), 200)
                offset = 0
                max_density = 0

                if hue_var is None:
                    hue_values = [None]
                else:
                    hue_values = data[hue_var].dropna().unique()

                for idx, hue_val in enumerate(hue_values):
                    subset = data[data[hue_var] == hue_val] if hue_var is not None else data
                    x_data = subset[var_x].dropna()
                    if len(x_data) < 2:
                        continue
                    try:
                        kde = gaussian_kde(x_data)
                        density = kde(x_values)
                        if cumulative:
                            density = np.cumsum(density)
                            density = density / density[-1]  # Normalize cumulative to 1
                        if common_norm and not cumulative:
                            density = density / density.max()
                        y_values = density + offset if multiple == "stack" else density
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode='lines',
                                fill='tozeroy' if multiple == "stack" else None,
                                name=str(hue_val) if hue_var else var_x,
                                line=dict(color=PALETTE[idx % len(PALETTE)])
                            )
                        )
                        if multiple == "stack":
                            offset += max_density
                    except Exception:
                        continue

                fig.update_layout(
                    title=f"1D Density Plot: {var_x}",
                    xaxis_title=var_x,
                    yaxis_title="Density",
                    width=800,
                    height=600,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

            # Density 2
            elif plot_type == 'density 2':
                import numpy as np

                x_cols = data.columns if risk_it_all else numeric_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                col_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)

                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue_var = st.sidebar.selectbox("Hue Variable", hue_cols, index=0)
                col_var = st.sidebar.selectbox("Facet Column", col_cols, index=0)

                kind = st.sidebar.selectbox("Kind", ["hist", "kde"])
                common_norm = st.sidebar.checkbox("Common Norm", value=False)
                rug = st.sidebar.checkbox("Add Rug Plot?", value=False)

                # Determine if faceting is active
                facet_active = col_var != 'None'

                if kind == "hist":
                    fig = px.density_heatmap(
                        data_frame=data,
                        x=var_x,
                        y=var_y,
                        color_continuous_scale='Viridis',
                        histnorm='probability density' if not common_norm else None,
                        facet_col=col_var if facet_active else None,
                        marginal_x='rug' if rug else None,
                        marginal_y='rug' if rug else None,
                        width=800,
                        height=600
                    )
                else:  # kind == "kde"
                    fig = px.density_contour(
                        data_frame=data,
                        x=var_x,
                        y=var_y,
                        color=hue_var if hue_var != 'None' else None,
                        color_discrete_sequence=PALETTE,
                        facet_col=col_var if facet_active else None,
                        marginal_x='rug' if rug else None,
                        marginal_y='rug' if rug else None,
                        width=800,
                        height=600
                    )

                st.plotly_chart(fig, use_container_width=True)

            # Scatter
            elif plot_type == 'scatter':
                import numpy as np
                # Define available columns depending on risk_it_all option
                x_cols = data.columns if risk_it_all else numeric_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = ['None'] + (data.columns if risk_it_all else categorical_cols)
                style_cols = ['None'] + (data.columns if risk_it_all else categorical_cols)
                size_cols = ['None'] + (data.columns if risk_it_all else numeric_cols)
                facet_cols = ['None'] + (categorical_cols if not risk_it_all else data.columns.tolist())

                # Sidebar selections
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0)
                style = st.sidebar.selectbox("Style (Symbol)", style_cols, index=0)
                size = st.sidebar.selectbox("Size (Bubble Size)", size_cols, index=0)
                facet_col = st.sidebar.selectbox("Facet Column", facet_cols, index=0)
                facet_row = st.sidebar.selectbox("Facet Row", facet_cols, index=0)

                alpha = st.sidebar.slider("Alpha (Opacity)", 0.0, 1.0, 0.8, 0.01)
                size_max = st.sidebar.slider("Max Marker Size", 5, 100, 10, 5)

                enhance_size = st.sidebar.selectbox(
                    "Enhance Size Differences",
                    options=["None", "Min-Max Normalize"],
                    index=0
                )

                plot_data = data.copy()

                # Size handling
                size_param = None
                if size != 'None':
                    if not pd.api.types.is_numeric_dtype(plot_data[size]):
                        st.warning(f"Size column '{size}' is not numeric. Size parameter will be ignored.")
                    else:
                        if plot_data[size].isna().any():
                            st.warning(f"Size column '{size}' contains NaN values. Dropping rows with NaN in size.")
                            plot_data = plot_data.dropna(subset=[size])
                        if (plot_data[size] < 0).any():
                            st.warning(f"Size column '{size}' contains negative values. Converting to absolute values.")
                            plot_data[size] = plot_data[size].abs()

                        if enhance_size == "Min-Max Normalize":
                            size_min = plot_data[size].min()
                            size_max_val = plot_data[size].max()
                            if size_max_val > size_min:
                                plot_data['size_for_plot'] = 10 + 40 * (plot_data[size] - size_min) / (size_max_val - size_min)
                            else:
                                plot_data['size_for_plot'] = 30
                            size_param = 'size_for_plot'
                        else:
                            size_param = size

                # Build scatter plot arguments
                scatter_kwargs = dict(
                    data_frame=plot_data,
                    x=var_x,
                    y=var_y,
                    opacity=alpha,
                    color_discrete_sequence=PALETTE,
                    width=800,
                    height=600,
                )

                if hue != 'None':
                    scatter_kwargs['color'] = hue
                if style != 'None':
                    scatter_kwargs['symbol'] = style
                if facet_col != 'None':
                    scatter_kwargs['facet_col'] = facet_col
                if facet_row != 'None':
                    scatter_kwargs['facet_row'] = facet_row

                # Apply size settings
                if size_param is not None:
                    scatter_kwargs['size'] = size_param
                    scatter_kwargs['size_max'] = size_max
                else:
                    # No size column selected: fix marker size manually
                    scatter_kwargs['size'] = np.full(len(plot_data), 1)  # dummy same size
                    scatter_kwargs['size_max'] = size_max

                fig = px.scatter(**scatter_kwargs)
                st.plotly_chart(fig, use_container_width=True)

            # Catplot
            elif plot_type == 'catplot':
                import numpy as np

                x_cols = data.columns if risk_it_all else categorical_cols
                y_cols = data.columns if risk_it_all else categorical_cols
                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                col_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                size_cols = ['None'] + (data.columns.tolist() if risk_it_all else numeric_cols)

                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue", hue_cols, index=0)
                col = st.sidebar.selectbox("Facet Column", col_cols, index=0)
                size_var = st.sidebar.selectbox("Size Variable", size_cols, index=0)

                kind = st.sidebar.selectbox("Kind", ["strip", "swarm"])
                facet = st.sidebar.checkbox("Facet", value=False)

                # Global point size if no size variable selected
                global_point_size = st.sidebar.slider("Global Marker Size", 5, 50, 10, 5)

                plot_data = data.copy()

                fig = px.strip(
                    plot_data,
                    x=var_x,
                    y=var_y,
                    color=hue if hue != 'None' else None,
                    facet_col=col if (facet and col != 'None') else None,
                    color_discrete_sequence=PALETTE,
                    width=800,
                    height=600
                )

                if size_var != 'None' and pd.api.types.is_numeric_dtype(plot_data[size_var]):
                    # Use size variable if selected
                    sizes = plot_data[size_var]
                    fig.update_traces(marker=dict(size=sizes, sizemode='diameter', sizeref=2.*max(sizes)/(15.**2), sizemin=4))
                else:
                    # Otherwise use a global constant size
                    fig.update_traces(marker=dict(size=global_point_size))

                if kind == "swarm":
                    fig.update_traces(jitter=0.3)

                st.plotly_chart(fig, use_container_width=True)

            # Regression
            elif plot_type == 'regression':
                import numpy as np
                import statsmodels.api as sm
                import plotly.graph_objects as go

                x_cols = data.columns if risk_it_all else numeric_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)

                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0)
                order = st.sidebar.selectbox("Order", [1, 2, 3], index=0)
                ci_level = st.sidebar.selectbox("Confidence Interval (%)", [68, 90, 95, 99], index=2)
                use_hue = st.sidebar.checkbox("Use Hue?", value=True)

                plot_data = data.copy()

                fig = go.Figure()

                # Define groups (by hue if used)
                if use_hue and hue != 'None':
                    groups = plot_data[hue].dropna().unique()
                else:
                    groups = [None]

                for idx, g in enumerate(groups):
                    if g is not None:
                        subset = plot_data[plot_data[hue] == g]
                    else:
                        subset = plot_data

                    X = subset[var_x]
                    Y = subset[var_y]

                    # Drop NaN
                    mask = (~X.isna()) & (~Y.isna())
                    X = X[mask]
                    Y = Y[mask]

                    if X.empty or Y.empty:
                        continue

                    # Scatter points
                    fig.add_trace(go.Scatter(
                        x=X,
                        y=Y,
                        mode='markers',
                        marker=dict(color=PALETTE[idx % len(PALETTE)], size=5),
                        name=str(g) if g is not None else "Data",
                        showlegend=True
                    ))

                    # Regression model
                    X_design = sm.add_constant(np.vander(X, N=order+1, increasing=True))
                    model = sm.OLS(Y, X_design).fit()

                    # X prediction points
                    x_pred = np.linspace(X.min(), X.max(), 100)
                    X_pred_design = sm.add_constant(np.vander(x_pred, N=order+1, increasing=True))

                    # Predict y values + confidence intervals
                    y_pred = model.predict(X_pred_design)
                    pred_summary = model.get_prediction(X_pred_design).summary_frame(alpha=(1 - ci_level / 100))

                    # Add regression line
                    fig.add_trace(go.Scatter(
                        x=x_pred,
                        y=y_pred,
                        mode='lines',
                        line=dict(color=PALETTE[idx % len(PALETTE)]),
                        name=f"Fit: {g}" if g is not None else "Fit",
                        showlegend=True
                    ))

                    # Add confidence band
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x_pred, x_pred[::-1]]),
                        y=np.concatenate([pred_summary['mean_ci_upper'], pred_summary['mean_ci_lower'][::-1]]),
                        fill='toself',
                        fillcolor=f"rgba(0, 100, 80, 0.2)",
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False
                    ))

                fig.update_layout(
                    title="Regression Plot with Confidence Interval",
                    xaxis_title=var_x,
                    yaxis_title=var_y,
                    width=800,
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

    # Render the plot if data is available
    if not data.empty:
        render_plot()
    else:
        st.warning("The dataset is empty after preprocessing. Please check your data or selections in the Home section.")

# --- MAIN ROUTING ---
def main():
    section = st.sidebar.radio("Select Section", ["Home", "Missing Data Imputation", "Visualization"])

    if section == "Home":
        home()
    elif section == "Missing Data Imputation":
        missing_data_imputation()
    elif section == "Visualization":
        visualization()

# --- Run the app ---
if __name__ == "__main__":
    main()