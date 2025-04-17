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
        df = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded!")

        # Display column info
        st.subheader("🔍 Column Overview")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": [df[col].dtype for col in df.columns],
            "Unique Values": [df[col].nunique() for col in df.columns]
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
            df[col] = df[col].astype(str)  # Or 'category' if preferred

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

# --- MISSING DATA IMPUTATION SECTION ---
def missing_data_imputation():
    st.subheader("🔄 Missing Data Imputation")

    # Check if filtered data exists
    if "filtered_df" not in st.session_state:
        st.warning("Please upload and process a dataset in the Home section first.")
        return

    # Retrieve filtered data
    df = st.session_state["filtered_df"]
    numeric_cols = st.session_state["numeric_cols"]
    categorical_cols = st.session_state["categorical_cols"]

    # Calculate missing data statistics
    st.subheader("📉 Missing Data Overview")
    missing_data = df.isna().sum()
    missing_percentage = (df.isna().sum() / len(df) * 100).round(2)
    missing_info = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_data,
        "Percentage Missing (%)": missing_percentage
    })
    st.dataframe(missing_info)

    # Check if there are any missing values
    if missing_data.sum() == 0:
        st.info("No missing values found in the dataset.")
        # Ensure is_imputed is False if no imputation is needed
        st.session_state["is_imputed"] = False
        return

    # KNN Imputation options
    st.subheader("🔧 Imputation Settings")
    impute_button = st.button("Impute Missing Values with KNN Imputer")

    if impute_button:
        # Separate numeric and categorical data
        numeric_df = df[numeric_cols].copy()
        categorical_df = df[categorical_cols].copy()

        # Apply KNN Imputation to numeric columns
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")
            imputed_numeric = imputer.fit_transform(numeric_df)
            imputed_numeric_df = pd.DataFrame(imputed_numeric, columns=numeric_cols, index=numeric_df.index)
        else:
            imputed_numeric_df = pd.DataFrame(index=df.index)

        # Combine imputed numeric data with categorical data
        imputed_df = pd.concat([imputed_numeric_df, categorical_df], axis=1)

        # Ensure column order matches original dataframe
        imputed_df = imputed_df[df.columns]

        # Store the imputed dataframe in session state
        st.session_state["imputed_df"] = imputed_df
        st.session_state["is_imputed"] = True

        st.success("Missing values have been imputed using KNN Imputer!")
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
        st.write("Using imputed dataset for visualization.")
    else:
        data = st.session_state["filtered_df"]
        st.write("Using original dataset for visualization.")

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
            data_c = data.drop(data.columns[data.nunique() == 1], axis=1)
            corr_matrix = data_c.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis', 
                           title="Correlation Matrix", width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "clustermap":
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

        elif plot_type == "pairplot":
            hue_var = st.sidebar.selectbox("Hue Variable", categorical_cols, index=0)
            numeric_cols_list = list(data.select_dtypes(include='number').columns)
            if not numeric_cols_list:
                st.warning("No numeric columns available for pairplot.")
                return
            
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
                        for k, hue_val in enumerate(data[hue_var].unique()):
                            subset = data[data[hue_var] == hue_val][col_i].dropna()
                            if len(subset) < 2:
                                continue
                            try:
                                kde = gaussian_kde(subset)
                                x_values = np.linspace(min(subset), max(subset), 100)
                                density = kde(x_values)
                                density = density / density.max()  # Normalize
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_values,
                                        y=density,
                                        mode='lines',
                                        name=f"{hue_val}",
                                        line=dict(color=PALETTE[k % len(PALETTE)]),
                                        showlegend=(i == 0 and j == 0)
                                    ),
                                    row=i+1, col=j+1
                                )
                            except Exception as e:
                                print(f"KDE error for {col_i}, {hue_val}: {e}")
                                continue
                    # Off-diagonal: Scatter (both upper and lower triangles)
                    else:
                        for k, hue_val in enumerate(data[hue_var].unique()):
                            subset = data[data[hue_var] == hue_val]
                            subset = subset[[col_j, col_i]].dropna()
                            if subset.empty:
                                continue
                            fig.add_trace(
                                go.Scatter(
                                    x=subset[col_j],
                                    y=subset[col_i],
                                    mode='markers',
                                    name=f"{hue_val}",
                                    marker=dict(
                                        color=PALETTE[k % len(PALETTE)],
                                        size=5
                                    ),
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
                x_cols = data.columns if risk_it_all else categorical_cols
                hue_cols = data.columns if risk_it_all else categorical_cols
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                hue = st.sidebar.selectbox("Hue", hue_cols, index=0)
                tplot = st.sidebar.selectbox("Plot Type", ["bars", "heatmap"])
                if tplot == "bars":
                    fig = px.histogram(data, x=var_x, color=hue, barmode='group', 
                                     color_discrete_sequence=PALETTE, width=800, height=600)
                    fig.update_traces(texttemplate='%{y}', textposition='auto')
                else:
                    df_2dhist = pd.DataFrame({
                        x_label: grp[var_x].value_counts()
                        for x_label, grp in data.groupby(hue)
                    }).fillna(0)
                    fig = px.imshow(df_2dhist, text_auto='.0f', color_continuous_scale='Viridis', 
                                   title=f"Heatmap: {hue} vs {var_x}", width=800, height=600)
                    fig.update_xaxes(title=hue)
                    fig.update_yaxes(title=var_x)
                st.plotly_chart(fig, use_container_width=True)

            # Boxes
            if plot_type == 'boxes':
                x_cols = data.columns if risk_it_all else categorical_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = data.columns if risk_it_all else categorical_cols
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue", hue_cols, index=0)
                tplot = st.sidebar.selectbox("Plot Type", ["boxplot", "lineplot", "violin"])
                no_hue = st.sidebar.checkbox("No Hue", value=False)
                hue_param = None if (no_hue and not risk_it_all) else hue
                if tplot == "boxplot":
                    fig = px.box(data, x=var_x, y=var_y, color=hue_param, 
                                color_discrete_sequence=PALETTE, width=800, height=600)
                elif tplot == "violin":
                    fig = px.violin(data, x=var_x, y=var_y, color=hue_param, box=True, 
                                   color_discrete_sequence=PALETTE, width=800, height=600)
                elif tplot == "lineplot":
                    fig = px.line(data.groupby([var_x, hue_param] if hue_param else var_x)[var_y]
                                 .mean().reset_index(), 
                                 x=var_x, y=var_y, color=hue_param, 
                                 color_discrete_sequence=PALETTE, width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)

            # Ridges
            elif plot_type == 'ridges':
                x_cols = data.columns if risk_it_all else categorical_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = data.columns if risk_it_all else categorical_cols
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue_var = st.sidebar.selectbox("Hue Variable", hue_cols, index=0)
                no_hue = st.sidebar.checkbox("No Hue", value=False)
                hue_param = var_x if no_hue else hue_var

                # Get unique var_x values for faceting
                x_values_unique = data[var_x].unique()
                n_rows = len(x_values_unique)
                if n_rows == 0:
                    st.warning("No valid categories in X variable.")
                    return

                # Create figure with subplots (one row per var_x value)
                fig = make_subplots(
                    rows=n_rows, cols=1,
                    row_titles=[str(x) for x in x_values_unique],
                    shared_xaxes=True,
                    vertical_spacing=0.1
                )
                x_range = [data[var_y].min(), data[var_y].max()]
                x_values = np.linspace(x_range[0], x_range[1], 100)

                # Plot KDE for each var_x and hue combination
                for row_idx, x_val in enumerate(x_values_unique, 1):
                    subset = data[data[var_x] == x_val]
                    hue_values = subset[hue_param].unique() if hue_param != var_x else [x_val]
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

                # Update layout for faceting
                fig.update_layout(
                    title=f"Ridge Plot Faceted by {var_x}",
                    width=800,
                    height=150 * n_rows,  # Adjust height based on number of facets
                    showlegend=True
                )
                # Update x-axes and y-axes
                fig.update_xaxes(title=var_y, row=n_rows, col=1)
                for row in range(1, n_rows + 1):
                    fig.update_yaxes(title="Density", range=[0, 0.5], row=row, col=1, showticklabels=False)
                
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

            # Density 1
            elif plot_type == 'density 1':
                x_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = data.columns if risk_it_all else categorical_cols
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                hue_var = st.sidebar.selectbox("Hue Variable", hue_cols, index=0)
                multiple = st.sidebar.selectbox("Multiple", ["layer", "stack"])
                common_norm = st.sidebar.checkbox("Common Norm", value=False)
                cumulative = st.sidebar.checkbox("Cumulative", value=False)
                
                fig = go.Figure()
                x_values = np.linspace(data[var_x].min(), data[var_x].max(), 100)
                offset = 0
                max_density = 0
                
                for idx, hue_val in enumerate(data[hue_var].unique() if hue_var else [None]):
                    subset = data[data[hue_var] == hue_val] if hue_var else data
                    x_data = subset[var_x].dropna()
                    if len(x_data) < 2:
                        continue
                    try:
                        kde = gaussian_kde(x_data)
                        density = kde(x_values)
                        if cumulative:
                            density = np.cumsum(density) / np.sum(density)
                        if common_norm:
                            density = density / density.max()
                        max_density = max(max_density, density.max())
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
                    title="1D Density Plot",
                    xaxis_title=var_x,
                    yaxis_title="Density",
                    width=800,
                    height=600,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

            # Density 2
            elif plot_type == 'density 2':
                x_cols = data.columns if risk_it_all else numeric_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = data.columns if risk_it_all else categorical_cols
                col_cols = data.columns if risk_it_all else categorical_cols
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue_var = st.sidebar.selectbox("Hue Variable", hue_cols, index=0)
                col_var = st.sidebar.selectbox("Col Variable", col_cols, index=0)
                kind = st.sidebar.selectbox("Kind", ["hist", "kde"])
                common_norm = st.sidebar.checkbox("Common Norm", value=False)
                rug = st.sidebar.checkbox("Rug", value=False)
                facet = st.sidebar.checkbox("Facet", value=False)
                if kind == "hist":
                    fig = px.density_heatmap(data, x=var_x, y=var_y, 
                                            color_continuous_scale='Viridis',
                                            facet_col=col_var if facet else None,
                                            histnorm='probability density' if not common_norm else None,
                                            marginal_x='rug' if rug else None,
                                            marginal_y='rug' if rug else None,
                                            width=800, height=600)
                else:
                    fig = px.density_contour(data, x=var_x, y=var_y, 
                                            color=hue_var, 
                                            facet_col=col_var if facet else None,
                                            marginal_x='rug' if rug else None,
                                            marginal_y='rug' if rug else None,
                                            color_discrete_sequence=PALETTE,
                                            width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)

            # Scatter
            elif plot_type == 'scatter':
                x_cols = data.columns if risk_it_all else numeric_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = data.columns if risk_it_all else categorical_cols
                style_cols = data.columns if risk_it_all else categorical_cols
                size_cols = data.columns if risk_it_all else numeric_cols
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue", hue_cols, index=0)
                style = st.sidebar.selectbox("Style", style_cols, index=0)
                size = st.sidebar.selectbox("Size", size_cols, index=0)
                alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.5, 0.01)
                size_max = st.sidebar.slider("Max Marker Size", 5, 50, 5, 5)
                use_style = st.sidebar.checkbox("Use Style", value=False)

                # Validate and preprocess the size column
                plot_data = data.copy()
                if not pd.api.types.is_numeric_dtype(plot_data[size]):
                    st.warning(f"Size column '{size}' is not numeric. Size parameter will be ignored.")
                    size_param = None
                else:
                    # Handle NaN and negative values
                    if plot_data[size].isna().any():
                        st.warning(f"Size column '{size}' contains NaN values. Dropping rows with NaN in size.")
                        plot_data = plot_data.dropna(subset=[size])
                    # Ensure non-negative values
                    if (plot_data[size] < 0).any():
                        st.warning(f"Size column '{size}' contains negative values. Converting to absolute values.")
                        plot_data[size] = plot_data[size].abs()
                    # Check for low variance
                    if plot_data[size].std() < 1e-6:
                        st.warning(f"Size column '{size}' has very low variance. Size differences may not be visible.")
                    # Normalize size values to improve visibility
                    size_min, size_max_val = plot_data[size].min(), plot_data[size].max()
                    if size_max_val > size_min:
                        plot_data[size] = 10 + 40 * (plot_data[size] - size_min) / (size_max_val - size_min)  # Scale to range [10, 50]
                    else:
                        st.warning(f"Size column '{size}' has no variation. All points will have the same size.")
                    size_param = size

                # Create scatter plot
                fig = px.scatter(
                    plot_data,
                    x=var_x,
                    y=var_y,
                    color=hue,
                    size=size_param,
                    size_max=size_max,
                    symbol=style if use_style else None,
                    opacity=alpha,
                    color_discrete_sequence=PALETTE,
                    width=800,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

            # Catplot
            elif plot_type == 'catplot':
                x_cols = data.columns if risk_it_all else categorical_cols
                y_cols = data.columns if risk_it_all else categorical_cols
                hue_cols = data.columns if risk_it_all else categorical_cols
                col_cols = data.columns if risk_it_all else categorical_cols
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue", hue_cols, index=0)
                col = st.sidebar.selectbox("Col", col_cols, index=0)
                kind = st.sidebar.selectbox("Kind", ["strip", "swarm"])
                facet = st.sidebar.checkbox("Facet", value=False)
                fig = px.strip(data, x=var_x, y=var_y, color=hue, 
                              facet_col=col if facet else None, 
                              color_discrete_sequence=PALETTE, width=800, height=600)
                if kind == "swarm":
                    fig.update_traces(jitter=0.3)
                st.plotly_chart(fig, use_container_width=True)

            # Regression
            elif plot_type == 'regression':
                x_cols = data.columns if risk_it_all else numeric_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = data.columns if risk_it_all else categorical_cols
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue", hue_cols, index=0)
                order = st.sidebar.selectbox("Order", [1, 2, 3])
                ci = st.sidebar.selectbox("Confidence Interval", [68, 95, 99, 0])
                use_hue = st.sidebar.checkbox("Use Hue", value=True)
                fig = px.scatter(data, x=var_x, y=var_y, color=hue if use_hue else None, 
                                trendline="ols" if order == 1 else "lowess", 
                                color_discrete_sequence=PALETTE, width=800, height=600)
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