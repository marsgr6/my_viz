import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.cluster import hierarchy
import warnings

warnings.filterwarnings('ignore')

def visualization():
    st.subheader("🧪 Visualization: Interactive Data Exploration")

    # Check if filtered data exists in session state
    if "filtered_df" not in st.session_state:
        st.warning("Please upload and process a dataset in the Home section first.")
        return

    # Use filtered_df as the dataset for visualization
    data = st.session_state["filtered_df"]
    # Check if imputation has been applied based on operation history
    if "operations" in st.session_state and any(op["type"] == "impute" for op in st.session_state["operations"]):
        st.success("Using dataset with imputed values for visualization.")
    #else:
        #st.info("Using original dataset for visualization.")

    # Check if current_numeric_cols and current_categorical_cols exist
    if "current_numeric_cols" not in st.session_state or "current_categorical_cols" not in st.session_state:
        st.warning("Column type information is missing. Please assign variable types in the Home section.")
        return

    numeric_cols = st.session_state["current_numeric_cols"]
    categorical_cols = st.session_state["current_categorical_cols"]

    # Convert categorical columns to strings
    for col in data.select_dtypes(include='category').columns:
        data[col] = data[col].astype('str')

    # Define plot types
    PLOT_TYPES = [
        'bars', 'boxes', 'lineplot', 'histogram', 'density 1', 'density 2',
        'scatter', 'catplot', 'missingno', 'correlation', 'clustermap',
        'pairplot', 'regression', "heatmap", 'ridges'
    ]

    if not numeric_cols:
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

    # ... (rest of the visualization section code remains unchanged)

    # Plot rendering function
    def render_plot():
        # Non-interactive plot types
        if plot_type == "correlation":
            import numpy as np
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd

            data_c = data.drop(data.columns[data.nunique() == 1], axis=1)
            corr_matrix = data_c.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis', 
                           title="Correlation Matrix", width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "clustermap":
            import numpy as np
            import plotly.graph_objects as go
            import pandas as pd

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
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from scipy.stats import gaussian_kde
            import pandas as pd


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
            import numpy as np
            import plotly.express as px
            import plotly.graph_objects as go
            from scipy.cluster.hierarchy import linkage, dendrogram
            import scipy.spatial.distance as ssd
            import pandas as pd

            st.sidebar.subheader("Missing Data Visualization")
            tplot = st.sidebar.selectbox("Plot Type", ["matrix", "bars", "heatmap"])

            # --- Build the missingness map (Presence = 1, Missing = 0) ---
            mapped_data = (~data.isna()).astype(int)

            # --- MATRIX ---
            if tplot == "matrix":
                fig = px.imshow(
                    mapped_data.values,
                    labels=dict(x="Columns", y="Rows", color="Present"),
                    x=mapped_data.columns,
                    y=mapped_data.index,
                    #color_continuous_scale=[[0.0, "yellow"], [1.0, "black"]],
                    color_continuous_scale="viridis",
                    range_color=[0, 1],  # Force 0/1
                    aspect='auto',
                    width=1000,
                    height=600
                )
                fig.update_layout(title="Missing Data Matrix (1 = Present, 0 = Missing)")
                st.plotly_chart(fig, use_container_width=True)

            # --- BARS ---
            elif tplot == "bars":
                import plotly.express as px

                # Build 0/1 map
                mapped_data = data.notnull().astype(int)

                # Calculate counts
                missing_counts = mapped_data.sum(axis=0)

                # --- Checkbox to filter full columns ---
                hide_no_missing = st.sidebar.checkbox("Hide Variables Without Missing Values", value=False)

                if hide_no_missing:
                    missing_counts = missing_counts[missing_counts < mapped_data.shape[0]]

                if missing_counts.empty:
                    st.warning("No variables with missing data to plot.")
                    st.stop()

                # --- Sort variables by counts ---
                missing_counts = missing_counts.sort_values(ascending=False)

                df_missing = pd.DataFrame({
                    "Variable": missing_counts.index,
                    "NonMissingCount": missing_counts.values
                })

                # --- Plot ---
                fig = px.bar(
                    df_missing,
                    x="Variable",
                    y="NonMissingCount",
                    color="NonMissingCount",
                    color_continuous_scale="Viridis",
                    text="NonMissingCount",
                    title="Non-Missing Data Count per Variable",
                    width=800,
                    height=600
                )

                fig.update_traces(textposition='outside')
                fig.update_layout(
                    xaxis_title="Variables",
                    yaxis_title="Count of Non-Missing Values",
                    margin=dict(l=80, r=20, t=50, b=150),
                    xaxis_tickangle=45,
                    coloraxis_colorbar=dict(title="NonMissingCount")
                )

                st.plotly_chart(fig, use_container_width=True)

            # --- HEATMAP (Missingness Correlation Matrix, sorted, lower triangle only) ---
            elif tplot == "heatmap":
                import numpy as np
                import plotly.express as px
                from scipy.spatial.distance import squareform
                from scipy.cluster.hierarchy import linkage, leaves_list

                # --- 1. Build the 0/1 map ---
                mapped_data = data.notnull().astype(int)

                # --- 2. Drop columns that are full (no missing values) ---
                cols_with_missing = mapped_data.columns[mapped_data.nunique() > 1]
                mapped_data = mapped_data[cols_with_missing]

                if mapped_data.shape[1] < 2:
                    st.warning("Not enough variables with missing data to plot a heatmap.")
                    st.stop()

                # --- 3. Correlation of missingness ---
                corr_matrix = mapped_data.corr()

                # --- 4. Force symmetry ---
                corr_matrix = (corr_matrix + corr_matrix.T) / 2

                # --- 5. Build condensed distance matrix ---
                distance_matrix = 1 - corr_matrix
                np.fill_diagonal(distance_matrix.values, 0)
                condensed_distance = squareform(distance_matrix.values)

                # --- 6. Perform hierarchical clustering ---
                linkage_matrix = linkage(condensed_distance, method='ward')
                ordered_indices = leaves_list(linkage_matrix)
                ordered_columns = corr_matrix.columns[ordered_indices]

                # --- 7. Sort the correlation matrix ---
                sorted_corr = corr_matrix.loc[ordered_columns, ordered_columns]

                # --- 8. Set NaN in upper triangle ---
                sorted_corr_masked = sorted_corr.copy()
                sorted_corr_masked.values[np.triu_indices_from(sorted_corr_masked, 1)] = np.nan

                # --- 9. Plot ---
                fig = px.imshow(
                    sorted_corr_masked,
                    text_auto=".2f",
                    color_continuous_scale="Viridis",
                    zmin=0,
                    zmax=1,
                    title="Lower Triangular Missingness Correlation Clustermap",
                    width=800,
                    height=800,
                    labels=dict(color="Correlation")
                )

                fig.update_layout(
                    xaxis_title="Variables",
                    yaxis_title="Variables",
                    xaxis_side="bottom",
                    yaxis_autorange="reversed"
                )

                st.plotly_chart(fig, use_container_width=True)

            elif tplot == "dendrogram":
                import numpy as np
                import plotly.figure_factory as ff
                import plotly.graph_objects as go
                from scipy.spatial.distance import squareform
                from scipy.cluster.hierarchy import linkage, leaves_list

                # --- 1. Build 0/1 map ---
                mapped_data = data.notnull().astype(int)

                # --- 2. Drop full columns (no missingness) ---
                cols_with_missing = mapped_data.columns[mapped_data.nunique() > 1]
                mapped_data = mapped_data[cols_with_missing]

                if mapped_data.shape[1] < 2:
                    st.warning("Not enough variables with missing data to plot a dendrogram.")
                    st.stop()

                # --- 3. Correlation of missingness ---
                corr_matrix = mapped_data.corr()

                # --- 4. Force symmetry
                corr_matrix = (corr_matrix + corr_matrix.T) / 2

                # --- 5. Distance matrix ---
                distance_matrix = 1 - corr_matrix
                np.fill_diagonal(distance_matrix.values, 0)
                condensed_distance = squareform(distance_matrix.values)

                # --- 6. Linkage ---
                linkage_matrix = linkage(condensed_distance, method="ward")
                ordered_indices = leaves_list(linkage_matrix)
                ordered_columns = corr_matrix.columns[ordered_indices]

                # --- 7. Reorder correlation matrix ---
                sorted_corr = corr_matrix.loc[ordered_columns, ordered_columns]

                # --- 8. Create vertical dendrogram ---
                dendro = ff.create_dendrogram(
                    mapped_data.T.values,
                    orientation='right',
                    labels=list(mapped_data.columns),
                    linkagefun=lambda _: linkage_matrix,
                    color_threshold=None
                )

                # --- 9. Plot heatmap separately ---
                heatmap = go.Heatmap(
                    z=sorted_corr.values,
                    x=ordered_columns,
                    y=ordered_columns,
                    colorscale='Viridis',
                    colorbar=dict(title="Missingness Corr"),
                    zmin=0,
                    zmax=1
                )

                # --- 10. Create figure with two subplots manually ---
                fig = go.Figure()

                # Add dendrogram traces
                for trace in dendro['data']:
                    fig.add_trace(trace)

                # Shift dendrogram traces to left to fit alongside heatmap
                for trace in fig['data']:
                    if trace['xaxis'] == 'x2':
                        trace['xaxis'] = 'x1'
                    if trace['yaxis'] == 'y2':
                        trace['yaxis'] = 'y1'

                fig.add_trace(heatmap)

                fig.update_layout(
                    width=1000,
                    height=1000,
                    showlegend=False,
                    xaxis=dict(
                        domain=[0.3, 1],
                        tickmode='array',
                        tickvals=list(range(len(ordered_columns))),
                        ticktext=ordered_columns,
                        tickangle=45
                    ),
                    yaxis=dict(
                        domain=[0, 0.7],
                        tickmode='array',
                        tickvals=list(range(len(ordered_columns))),
                        ticktext=ordered_columns[::-1],  # Flip y axis
                        autorange='reversed'
                    ),
                    xaxis2=dict(domain=[0, 0.2]),  # for dendrogram
                    yaxis2=dict(domain=[0.7, 1]),  # for dendrogram
                    margin=dict(l=100, t=50, b=100)
                )

                st.plotly_chart(fig, use_container_width=True)

        # Interactive plot types with risk_it_all toggle
        else:
            # Bars
            if plot_type == 'bars':
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots  # Moved to top to avoid UnboundLocalError
                import numpy as np
                import pandas as pd

                # Setup columns
                if categorical_cols:
                    x_cols = data.columns.tolist() if risk_it_all else categorical_cols
                else:
                    x_cols = numeric_cols

                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                facet_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)

                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                hue = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0)
                facet_col = st.sidebar.selectbox("Facet Column (optional)", facet_cols, index=0)
                facet_row = st.sidebar.selectbox("Facet Row (optional)", facet_cols, index=0)

                tplot = st.sidebar.selectbox("Plot Type", ["bars", "heatmap"])
                bar_mode = st.sidebar.selectbox("Bars Mode", ["Histogram", "Raw Values (height = value in order)"])
                # Add "Time is here" checkbox for bars
                time_series = st.sidebar.checkbox("Time is here", value=False) if tplot == "bars" else False
                var_y = None  # Initialize var_y to None
                if time_series:
                    y_cols = data.columns.tolist()
                    # Set a default numerical column if available, otherwise None
                    default_y = next((col for col in y_cols if pd.api.types.is_numeric_dtype(data[col])), None)
                    var_y = st.sidebar.selectbox("Y Variable (for Time Series)", y_cols, index=y_cols.index(default_y) if default_y else 0)

                plot_data = data.copy()

                # Fix None values
                hue = None if hue == 'None' else hue

                # Force hue to string
                if hue is not None:
                    plot_data[hue] = plot_data[hue].astype(str)

                # Custom Orders
                category_orders = {}

                # Order for X
                if var_x in plot_data.columns and plot_data[var_x].dtype.name in ['object', 'category']:
                    custom_order_x = st.sidebar.multiselect(
                        f"Custom Order for x {var_x}",
                        options=plot_data[var_x].dropna().unique().tolist(),
                        default=sorted(plot_data[var_x].dropna().unique().tolist())
                    )
                    plot_data[var_x] = pd.Categorical(plot_data[var_x], categories=custom_order_x, ordered=True)
                    category_orders[var_x] = custom_order_x

                # Order for Hue
                if hue != 'None' and hue in plot_data.columns and plot_data[hue].dtype.name in ['object', 'category']:
                    custom_order_hue = st.sidebar.multiselect(
                        f"Custom Order for hue {hue}",
                        options=plot_data[hue].dropna().unique().tolist(),
                        default=sorted(plot_data[hue].dropna().unique().tolist())
                    )
                    plot_data[hue] = pd.Categorical(plot_data[hue], categories=custom_order_hue, ordered=True)
                    category_orders[hue] = custom_order_hue

                # Order for Facet Column
                if facet_col != 'None' and facet_col in plot_data.columns and plot_data[facet_col].dtype.name in ['object', 'category']:
                    custom_order_facet_col = st.sidebar.multiselect(
                        f"Custom Order for Facet Col {facet_col}",
                        options=plot_data[facet_col].dropna().unique().tolist(),
                        default=sorted(plot_data[facet_col].dropna().unique().tolist())
                    )
                    plot_data[facet_col] = pd.Categorical(plot_data[facet_col], categories=custom_order_facet_col, ordered=True)
                    category_orders[facet_col] = custom_order_facet_col

                # Order for Facet Row
                if facet_row != 'None' and facet_row in plot_data.columns and plot_data[facet_row].dtype.name in ['object', 'category']:
                    custom_order_facet_row = st.sidebar.multiselect(
                        f"Custom Order for Facet Row {facet_row}",
                        options=plot_data[facet_row].dropna().unique().tolist(),
                        default=sorted(plot_data[facet_row].dropna().unique().tolist())
                    )
                    plot_data[facet_row] = pd.Categorical(plot_data[facet_row], categories=custom_order_facet_row, ordered=True)
                    category_orders[facet_row] = custom_order_facet_row

                # --- Bars Plot ---
                if tplot == "bars":
                    if time_series:
                        # Ensure var_y is selected
                        if var_y is None:
                            st.warning("Please select a Y variable for Time Series mode.")
                            return

                        # Validate var_x is object (e.g., date) and var_y is numerical
                        is_object = False
                        if isinstance(plot_data[var_x].dtype, pd.CategoricalDtype):
                            # For categorical columns, check the dtype of the categories
                            if plot_data[var_x].cat.categories.dtype.name == 'object':
                                is_object = True
                        else:
                            # For non-categorical columns, check the dtype directly
                            if plot_data[var_x].dtype.name == 'object':
                                is_object = True

                        if not is_object:
                            st.warning(f"Time series mode requires '{var_x}' to be an object column (e.g., dates).")
                            return

                        # Check if var_y is numerical
                        if not pd.api.types.is_numeric_dtype(plot_data[var_y]):
                            st.warning(f"Time series mode requires '{var_y}' to be a numerical column.")
                            return

                        # Determine facet values
                        facet_col_vals = plot_data[facet_col].dropna().unique() if facet_col != 'None' else [None]
                        facet_row_vals = plot_data[facet_row].dropna().unique() if facet_row != 'None' else [None]

                        n_cols = max(len(facet_col_vals), 1)
                        n_rows = max(len(facet_row_vals), 1)

                        # Create subplot figure
                        fig = make_subplots(
                            rows=n_rows,
                            cols=n_cols,
                            shared_yaxes=True,
                            shared_xaxes=True,
                            horizontal_spacing=0.1,
                            vertical_spacing=0.12,
                            subplot_titles=[
                                f"{facet_row}: {r} | {facet_col}: {c}"
                                if facet_row != 'None' and facet_col != 'None' else
                                f"{facet_col}: {c}" if facet_col != 'None' else
                                f"{facet_row}: {r}" if facet_row != 'None' else ""
                                for r in facet_row_vals for c in facet_col_vals
                            ]
                        )

                        # Color mapping for hue
                        if hue is not None:
                            hue_values = plot_data[hue].dropna().unique()
                        else:
                            hue_values = [None]

                        color_map = {}
                        for idx, hv in enumerate(hue_values):
                            color_map[hv] = PALETTE[idx % len(PALETTE)]

                        # Add bar traces for each facet and hue
                        for i_row, row_val in enumerate(facet_row_vals):
                            for i_col, col_val in enumerate(facet_col_vals):
                                row_idx = i_row + 1
                                col_idx = i_col + 1

                                # Filter data by facet values
                                sub_data = plot_data.copy()
                                if facet_row != 'None':
                                    sub_data = sub_data[sub_data[facet_row] == row_val]
                                if facet_col != 'None':
                                    sub_data = sub_data[sub_data[facet_col] == col_val]

                                if sub_data.empty:
                                    continue

                                for hue_val in hue_values:
                                    # Filter subset by hue
                                    subset = sub_data.copy()
                                    if hue_val is not None:
                                        subset = subset[subset[hue] == hue_val]

                                    if subset.empty:
                                        continue

                                    # Get bar color for this hue
                                    bar_color = color_map[hue_val]

                                    # Plot raw data as bars without sorting
                                    fig.add_trace(
                                        go.Bar(
                                            x=subset[var_x],
                                            y=subset[var_y],
                                            name=str(hue_val) if hue_val else "Bars",
                                            marker_color=bar_color,
                                            showlegend=(row_idx == 1 and col_idx == 1)
                                        ),
                                        row=row_idx,
                                        col=col_idx
                                    )

                        # Update layout
                        fig.update_layout(
                            width=800,
                            height=600,
                            title=f"Time Series Bar Plot: {var_y} vs {var_x}",
                            xaxis_title=var_x,
                            yaxis_title=var_y,
                            legend_title=hue if hue is not None else None,
                            showlegend=True,
                            barmode='group'
                        )

                    else:
                        if bar_mode == "Histogram":
                            if hue is not None:
                                fig = px.histogram(
                                    plot_data,
                                    x=var_x,
                                    color=hue,
                                    barmode='group',
                                    facet_col=facet_col if facet_col != 'None' else None,
                                    facet_row=facet_row if facet_row != 'None' else None,
                                    color_discrete_sequence=PALETTE,
                                    category_orders=category_orders,
                                    width=800,
                                    height=600
                                )
                            else:
                                fig = px.histogram(
                                    plot_data,
                                    x=var_x,
                                    facet_col=facet_col if facet_col != 'None' else None,
                                    facet_row=facet_row if facet_row != 'None' else None,
                                    color_discrete_sequence=PALETTE,
                                    category_orders=category_orders,
                                    width=800,
                                    height=600
                                )
                            fig.update_traces(texttemplate='%{y}', textposition='auto')

                        else:  # Raw Values
                            df_plot = plot_data.reset_index()
                            x_axis = 'index'
                            y_axis = var_x

                            if hue is not None:
                                fig = px.bar(
                                    df_plot,
                                    x=x_axis,
                                    y=y_axis,
                                    color=hue,
                                    facet_col=facet_col if facet_col != 'None' else None,
                                    facet_row=facet_row if facet_row != 'None' else None,
                                    color_discrete_sequence=PALETTE,
                                    category_orders=category_orders,
                                    width=800,
                                    height=600
                                )
                            else:
                                fig = px.bar(
                                    df_plot,
                                    x=x_axis,
                                    y=y_axis,
                                    facet_col=facet_col if facet_col != 'None' else None,
                                    facet_row=facet_row if facet_row != 'None' else None,
                                    color_discrete_sequence=PALETTE,
                                    category_orders=category_orders,
                                    width=800,
                                    height=600
                                )
                            fig.update_layout(xaxis_title="Index", yaxis_title=y_axis)

                else:  # --- Heatmap ---
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
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                import pandas as pd

                x_cols = data.columns if risk_it_all else categorical_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                facet_cols = ['None'] + (categorical_cols if not risk_it_all else data.columns.tolist())

                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0)
                facet_col = st.sidebar.selectbox("Facet Column (optional)", facet_cols, index=0)
                facet_row = st.sidebar.selectbox("Facet Row (optional)", facet_cols, index=0)
                tplot = st.sidebar.selectbox("Plot Type", ["boxplot", "violin"])

                # Fix None values
                hue_used = None if hue == 'None' else hue
                facet_col_used = None if facet_col == 'None' else facet_col
                facet_row_used = None if facet_row == 'None' else facet_row

                # Show swarm points checkbox only when:
                # 1. No faceting (facet_col_used and facet_row_used are None)
                # 2. hue_used is None or hue_used == var_x
                show_swarm_checkbox = (facet_col_used is None and facet_row_used is None and (hue_used is None or hue_used == var_x))
                swarm_points = st.sidebar.checkbox("Overlay Swarm of Points", value=False) if show_swarm_checkbox else False
                band_interval = st.sidebar.checkbox("Show Band in Lineplot (CI)", value=False)
                # Add "Time is here" checkbox for lineplot
                time_series = st.sidebar.checkbox("Time is here", value=False) if tplot == "lineplot" else False
                # Add CI level selector for lineplot (only when time_series is not checked)
                ci_level = st.sidebar.selectbox("Confidence Interval Level", ["68%", "95%", "99%"], index=1) if (band_interval and not time_series) else "95%"
                ci_level_value = {"68%": 0.68, "95%": 0.95, "99%": 0.99}[ci_level]

                plot_data = data.copy()

                # Force Hue column to string if selected
                if hue_used is not None and hue_used in plot_data.columns:
                    plot_data[hue_used] = plot_data[hue_used].astype(str)

                # Let the user specify custom orders
                category_orders = {}

                if var_x in plot_data.columns and plot_data[var_x].dtype.name in ['object', 'category']:
                    custom_order_x = st.sidebar.multiselect(
                        f"Custom Order for x {var_x}",
                        options=plot_data[var_x].dropna().unique().tolist(),
                        default=sorted(plot_data[var_x].dropna().unique().tolist())
                    )
                    plot_data[var_x] = pd.Categorical(plot_data[var_x], categories=custom_order_x, ordered=True)
                    category_orders[var_x] = custom_order_x

                if hue_used and hue_used in plot_data.columns and plot_data[hue_used].dtype.name in ['object', 'category']:
                    custom_order_hue = st.sidebar.multiselect(
                        f"Custom Order for hue {hue_used}",
                        options=plot_data[hue_used].dropna().unique().tolist(),
                        default=sorted(plot_data[hue_used].dropna().unique().tolist())
                    )
                    plot_data[hue_used] = pd.Categorical(plot_data[hue_used], categories=custom_order_hue, ordered=True)
                    category_orders[hue_used] = custom_order_hue

                if facet_col_used and facet_col_used in plot_data.columns and plot_data[facet_col_used].dtype.name in ['object', 'category']:
                    custom_order_facet_col = st.sidebar.multiselect(
                        f"Custom Order for col {facet_col_used}",
                        options=plot_data[facet_col_used].dropna().unique().tolist(),
                        default=sorted(plot_data[facet_col_used].dropna().unique().tolist())
                    )
                    plot_data[facet_col_used] = pd.Categorical(plot_data[facet_col_used], categories=custom_order_facet_col, ordered=True)
                    category_orders[facet_col_used] = custom_order_facet_col

                if facet_row_used and facet_row_used in plot_data.columns and plot_data[facet_row_used].dtype.name in ['object', 'category']:
                    custom_order_facet_row = st.sidebar.multiselect(
                        f"Custom Order for row {facet_row_used}",
                        options=plot_data[facet_row_used].dropna().unique().tolist(),
                        default=sorted(plot_data[facet_row_used].dropna().unique().tolist())
                    )
                    plot_data[facet_row_used] = pd.Categorical(plot_data[facet_row_used], categories=custom_order_facet_row, ordered=True)
                    category_orders[facet_row_used] = custom_order_facet_row

                # --- Build the plot ---
                if tplot in ["boxplot", "violin"]:
                    plot_kwargs = dict(
                        data_frame=plot_data,
                        y=var_y,
                        color_discrete_sequence=PALETTE,
                        width=800,
                        height=600,
                        category_orders=category_orders
                    )

                    if var_x:
                        plot_kwargs['x'] = var_x
                    if hue_used:
                        plot_kwargs['color'] = hue_used
                    if facet_col_used:
                        plot_kwargs['facet_col'] = facet_col_used
                    if facet_row_used:
                        plot_kwargs['facet_row'] = facet_row_used

                    if tplot == "boxplot":
                        fig = px.box(**plot_kwargs)
                    elif tplot == "violin":
                        #stop_at_min_max = st.sidebar.checkbox("Stop Violin Density at Min/Max", value=False) if tplot == "violin" else False
                        #plot_kwargs['cut'] = 0 if stop_at_min_max else None
                        fig = px.violin(**plot_kwargs, box=True)

                    # Swarm Points (only when hue_used is None or hue_used == var_x, and no faceting)
                    if swarm_points:
                        # Use Plotly Express to generate swarm points with consistent hue coloring
                        scatter_kwargs = dict(
                            data_frame=plot_data,
                            x=var_x,
                            y=var_y,
                            color=hue_used,
                            color_discrete_sequence=PALETTE,
                            category_orders=category_orders
                        )
                        scatter_fig = px.scatter(**scatter_kwargs)

                        # Add scatter traces to the main figure
                        for trace in scatter_fig.data:
                            trace.update(
                                mode='markers',
                                marker=dict(size=5, opacity=0.6, line=dict(width=0)),
                                showlegend=False
                            )
                            # Since faceting is disabled (due to show_swarm_checkbox), add to main plot
                            fig.add_trace(trace, row=1, col=1)

                    # Render plot for boxplot or violin
                    plot_placeholder = st.empty()
                    plot_placeholder.plotly_chart(fig, use_container_width=True, key=f"{tplot}_{var_x}_{var_y}_{hue_used}")

            # --- Lineplot ---
            elif plot_type == 'lineplot':
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                import pandas as pd
                from plotly.subplots import make_subplots
                import re
                from scipy.stats import t

                # Assuming PALETTE, data, risk_it_all, categorical_cols, numeric_cols, and st are defined elsewhere

                x_cols = data.columns if risk_it_all else categorical_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                facet_cols = ['None'] + (categorical_cols if not risk_it_all else data.columns.tolist())
                style_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                width_cols = ['None'] + (data.columns.tolist() if risk_it_all else numeric_cols)

                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0)
                facet_col = st.sidebar.selectbox("Facet Column (optional)", facet_cols, index=0)
                facet_row = st.sidebar.selectbox("Facet Row (optional)", facet_cols, index=0)
                style = st.sidebar.selectbox("Line Style (Categorical)", style_cols, index=0)
                width = st.sidebar.selectbox("Line Width (Numeric)", width_cols, index=0)
                show_markers = st.sidebar.checkbox("Show Markers", value=True)
                band_interval = st.sidebar.checkbox("Show Band in Lineplot (CI)", value=False)
                time_series = st.sidebar.checkbox("Time is here", value=False)
                aggregation_method = st.sidebar.selectbox("Aggregation Method", ["Mean", "Sum"], index=0) if not time_series else "Raw"
                ci_level = st.sidebar.selectbox("Confidence Interval Level", ["68%", "95%", "99%"], index=1) if (band_interval and not time_series) else "95%"
                ci_level_value = {"68%": 0.68, "95%": 0.95, "99%": 0.99}[ci_level]

                # Fix None values
                hue_used = None if hue == 'None' else hue
                facet_col_used = None if facet_col == 'None' else facet_col
                facet_row_used = None if facet_row == 'None' else facet_row
                style_used = None if style == 'None' else style
                width_used = None if width == 'None' else width

                plot_data = data.copy()

                # Force Hue column to string if selected
                if hue_used is not None and hue_used in plot_data.columns:
                    plot_data[hue_used] = plot_data[hue_used].astype(str)

                # Let the user specify custom orders
                category_orders = {}

                if var_x in plot_data.columns and plot_data[var_x].dtype.name in ['object', 'category']:
                    custom_order_x = st.sidebar.multiselect(
                        f"Custom Order for x {var_x}",
                        options=plot_data[var_x].dropna().unique().tolist(),
                        default=sorted(plot_data[var_x].dropna().unique().tolist())
                    )
                    plot_data[var_x] = pd.Categorical(plot_data[var_x], categories=custom_order_x, ordered=True)
                    category_orders[var_x] = custom_order_x

                if hue_used and hue_used in plot_data.columns and plot_data[hue_used].dtype.name in ['object', 'category']:
                    custom_order_hue = st.sidebar.multiselect(
                        f"Custom Order for hue {hue_used}",
                        options=plot_data[hue_used].dropna().unique().tolist(),
                        default=sorted(plot_data[hue_used].dropna().unique().tolist())
                    )
                    plot_data[hue_used] = pd.Categorical(plot_data[hue_used], categories=custom_order_hue, ordered=True)
                    category_orders[hue_used] = custom_order_hue

                if facet_col_used and facet_col_used in plot_data.columns and plot_data[facet_col_used].dtype.name in ['object', 'category']:
                    custom_order_facet_col = st.sidebar.multiselect(
                        f"Custom Order for col {facet_col_used}",
                        options=plot_data[facet_col_used].dropna().unique().tolist(),
                        default=sorted(plot_data[facet_col_used].dropna().unique().tolist())
                    )
                    plot_data[facet_col_used] = pd.Categorical(plot_data[facet_col_used], categories=custom_order_facet_col, ordered=True)
                    category_orders[facet_col_used] = custom_order_facet_col

                if facet_row_used and facet_row_used in plot_data.columns and plot_data[facet_row_used].dtype.name in ['object', 'category']:
                    custom_order_facet_row = st.sidebar.multiselect(
                        f"Custom Order for row {facet_row_used}",
                        options=plot_data[facet_row_used].dropna().unique().tolist(),
                        default=sorted(plot_data[facet_row_used].dropna().unique().tolist())
                    )
                    plot_data[facet_row_used] = pd.Categorical(plot_data[facet_row_used], categories=custom_order_facet_row, ordered=True)
                    category_orders[facet_row_used] = custom_order_facet_row

                # Determine facet values
                facet_col_vals = plot_data[facet_col_used].dropna().unique().tolist() if facet_col_used else [None]
                facet_row_vals = plot_data[facet_row_used].dropna().unique().tolist() if facet_row_used else [None]

                n_cols = len(facet_col_vals) if facet_col_used else 1
                n_rows = len(facet_row_vals) if facet_row_used else 1

                # Create subplot figure
                fig = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    shared_yaxes=True,
                    shared_xaxes=True,
                    horizontal_spacing=0.1,
                    vertical_spacing=0.12,
                    subplot_titles=[
                        f"{facet_row_used}: {r} | {facet_col_used}: {c}"
                        if facet_row_used and facet_col_used else
                        f"{facet_col_used}: {c}" if facet_col_used else
                        f"{facet_row_used}: {r}" if facet_row_used else ""
                        for r in facet_row_vals for c in facet_col_vals
                    ]
                )

                # Color mapping for hue (use hue if different from var_x, otherwise single color)
                if hue_used and hue_used != var_x:
                    hue_values = plot_data[hue_used].dropna().unique()
                else:
                    hue_values = [None]  # Treat as no hue for coloring

                color_map = {}
                for idx, hv in enumerate(hue_values):
                    color_map[hv] = PALETTE[idx % len(PALETTE)]

                # Define line styles based on style column (default to solid if None or not categorical)
                line_styles = {
                    'solid': 'solid',
                    'dash': 'dash',
                    'dot': 'dot',
                    'dashdot': 'dashdot'
                }
                if style_used and style_used in plot_data.columns and plot_data[style_used].dtype.name in ['object', 'category']:
                    style_values = plot_data[style_used].dropna().unique()
                    style_map = {v: list(line_styles.values())[i % len(line_styles)] for i, v in enumerate(style_values)}
                else:
                    style_map = {None: 'solid'}

                # Define line widths based on width column (default to 2 if None or not numeric)
                if width_used and width_used in plot_data.columns and np.issubdtype(plot_data[width_used].dtype, np.number):
                    width_map = plot_data[width_used].dropna().to_dict()
                else:
                    width_map = {None: 2}

                # Helper function to convert line color to semi-transparent fillcolor
                def to_fillcolor(line_color, alpha=0.2):
                    try:
                        # Handle hex colors
                        if line_color.startswith('#'):
                            from matplotlib.colors import to_rgb
                            rgb = to_rgb(line_color)
                            return f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},{alpha})'
                        
                        # Handle rgb strings (e.g., 'rgb(102,194,165)')
                        rgb_match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', line_color)
                        if rgb_match:
                            r, g, b = map(int, rgb_match.groups())
                            return f'rgba({r},{g},{b},{alpha})'
                        
                        # Handle named colors (e.g., 'blue', 'red')
                        from matplotlib.colors import to_rgb
                        rgb = to_rgb(line_color)
                        return f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},{alpha})'
                        
                    except (ValueError, KeyError):
                        # Fallback to a default color
                        return f'rgba(128,128,128,{alpha})'  # Gray fallback

                # Track which hue values have been added to the legend to avoid duplicates
                seen_hue_values = set()

                # Add traces for each facet and hue
                for i_row, row_val in enumerate(facet_row_vals):
                    for i_col, col_val in enumerate(facet_col_vals):
                        row_idx = i_row + 1
                        col_idx = i_col + 1

                        # Filter data by facet values
                        sub_data = plot_data.copy()
                        if facet_row_used:
                            sub_data = sub_data[sub_data[facet_row_used] == row_val]
                        if facet_col_used:
                            sub_data = sub_data[sub_data[facet_col_used] == col_val]

                        if sub_data.empty:
                            continue

                        for hue_val in hue_values:
                            # Filter subset by hue
                            subset = sub_data.copy()
                            if hue_used and hue_used != var_x:
                                subset = subset[subset[hue_used] == hue_val]

                            if subset.empty:
                                continue

                            # Get line color for this hue
                            line_color = color_map[hue_val]

                            # Determine if this hue value should show in the legend
                            show_in_legend = hue_val not in seen_hue_values
                            if show_in_legend:
                                seen_hue_values.add(hue_val)

                            if time_series:
                                # Validate var_x is object (e.g., date) and var_y is numerical
                                is_object = False
                                if isinstance(plot_data[var_x].dtype, pd.CategoricalDtype):
                                    if plot_data[var_x].cat.categories.dtype.name == 'object':
                                        is_object = True
                                else:
                                    if plot_data[var_x].dtype.name == 'object':
                                        is_object = True

                                if not is_object:
                                    st.warning(f"Time series mode requires '{var_x}' to be an object column (e.g., dates).")
                                    continue
                                if not np.issubdtype(plot_data[var_y].dtype, np.number):
                                    st.warning(f"Time series mode requires '{var_y}' to be a numerical column.")
                                    continue

                                # Plot raw time series data without sorting
                                scatter_kwargs = dict(
                                    x=subset[var_x],
                                    y=subset[var_y],
                                    mode='lines' + ('+markers' if show_markers else ''),
                                    name=str(hue_val) if hue_val else "Line",
                                    legendgroup=str(hue_val) if hue_val else None,
                                    marker=dict(size=6, color=line_color),
                                    line=dict(
                                        color=line_color,
                                        dash=style_map.get(subset[style_used].iloc[0] if style_used else None, 'solid'),
                                        width=width_map.get(subset[width_used].iloc[0] if width_used else None, 2)
                                    ),
                                    showlegend=show_in_legend
                                )
                            else:
                                # Prepare grouping columns and remove duplicates
                                group_cols = list(dict.fromkeys([col for col in [var_x, hue_used, facet_col_used, facet_row_used] if col is not None and col != var_x]))
                                group_cols = [var_x] + group_cols  # Ensure var_x is first

                                # Group by selected columns and aggregate mean, std, count, and sum
                                grouped = plot_data.groupby(group_cols)[var_y].agg(['mean', 'std', 'count', 'sum']).reset_index()

                                # Filter grouped data for this facet and hue
                                sub = grouped.copy()
                                if hue_used and hue_used != var_x:
                                    sub = sub[sub[hue_used] == hue_val]
                                if facet_row_used:
                                    sub = sub[sub[facet_row_used] == row_val]
                                if facet_col_used:
                                    sub = sub[sub[facet_col_used] == col_val]

                                if sub.empty or sub['count'].iloc[0] < 2:
                                    continue

                                # Calculate confidence intervals using t.interval with selected CI level
                                alpha = 1 - ci_level_value
                                sub['se'] = sub['std'] / np.sqrt(sub['count'])
                                sub['ci_lower'], sub['ci_upper'] = t.interval(
                                    ci_level_value,
                                    sub['count'] - 1,
                                    loc=sub['mean'],
                                    scale=sub['se']
                                )
                                sub['ci'] = (sub['ci_upper'] - sub['ci_lower']) / 2

                                # Select y-values based on aggregation method
                                y_values = sub['sum'] if aggregation_method == "Sum" else sub['mean']

                                # Plot line with error bars if band_interval is enabled
                                scatter_kwargs = dict(
                                    x=sub[var_x],
                                    y=y_values,
                                    mode='lines' + ('+markers' if show_markers else ''),
                                    name=str(hue_val) if hue_val else "Line",
                                    legendgroup=str(hue_val) if hue_val else None,
                                    marker=dict(size=6, color=line_color),
                                    line=dict(
                                        color=line_color,
                                        dash=style_map.get(sub[style_used].iloc[0] if style_used else None, 'solid'),
                                        width=width_map.get(sub[width_used].iloc[0] if width_used else None, 2)
                                    ),
                                    showlegend=show_in_legend
                                )
                                if band_interval and aggregation_method == "Mean":
                                    scatter_kwargs['error_y'] = dict(
                                        type='data',
                                        symmetric=True,
                                        array=sub['ci'],
                                        thickness=1.5,
                                        width=5,
                                        color=line_color
                                    )

                            fig.add_trace(go.Scatter(**scatter_kwargs), row=row_idx, col=col_idx)

                # Update layout
                fig.update_layout(
                    width=1000,
                    height=400 * n_rows,
                    title=f"{'Time Series ' if time_series else ''}Lineplot: {aggregation_method} of {var_y} vs {var_x}" + (f" with {ci_level} CI" if not time_series and aggregation_method == "Mean" else ""),
                    showlegend=True,
                    legend_title=hue_used if hue_used else None
                )

                # Add axis titles to each subplot
                if facet_row_used or facet_col_used:
                    for i in range(1, n_rows + 1):
                        for j in range(1, n_cols + 1):
                            fig.update_xaxes(title_text=var_x, row=i, col=j, showticklabels=True)
                            fig.update_yaxes(title_text=f"{aggregation_method} of {var_y}", row=i, col=j, showticklabels=True)

                # Render plot
                plot_placeholder = st.empty()
                plot_placeholder.plotly_chart(fig, use_container_width=True, key=f"lineplot_{var_x}_{var_y}_{hue_used}_{aggregation_method}")

            # --- Ridges sometime are nice ---
            elif plot_type == 'ridges':
                import numpy as np
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                from scipy.stats import gaussian_kde
                import pandas as pd
                import plotly.express as px

                # Swap column types: X should be numeric, Y should be categorical
                x_cols = data.columns if risk_it_all else numeric_cols
                y_cols = data.columns if risk_it_all else categorical_cols
                cat_cols = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

                only_numeric = len(cat_cols) == 0

                height = st.sidebar.slider("Height (each row)", 5, 500, 200, 5)

                plot_data = data.copy()

                if not only_numeric:
                    var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)

                    var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                    y_values_unique = plot_data[var_y].dropna().unique()

                    n_rows = len(y_values_unique)

                    # ⛔ If too many categories -> fallback
                    if n_rows > 100:
                        st.warning(f"Too many categories in {var_y} ({n_rows} unique values). Plotting numeric columns instead.")
                        only_numeric = True
                    else:
                        hue_cols = data.columns if risk_it_all else categorical_cols
                        hue = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0)
                        no_hue = st.sidebar.checkbox("No Hue?", value=False)

                        hue_param = var_y if no_hue else hue

                        # --- Sorting options ---
                        # Custom order for panels (Y var, now categorical)
                        y_values_unique = plot_data[var_y].dropna().unique()
                        custom_order_y = st.sidebar.multiselect(
                            f"Custom Order for y {var_y}",
                            options=sorted(y_values_unique.tolist()),
                            default=sorted(y_values_unique.tolist())
                        )
                        plot_data[var_y] = pd.Categorical(plot_data[var_y], categories=custom_order_y, ordered=True)
                        y_values_unique = custom_order_y

                        # Custom order for hues
                        if hue_param != var_y and hue_param in plot_data.columns:
                            hue_values_unique = plot_data[hue_param].dropna().unique()
                            custom_order_hue = st.sidebar.multiselect(
                                f"Custom Order for hue {hue_param}",
                                options=sorted(hue_values_unique.tolist()),
                                default=sorted(hue_values_unique.tolist())
                            )
                            plot_data[hue_param] = pd.Categorical(plot_data[hue_param], categories=custom_order_hue, ordered=True)
                            global_hue_categories = custom_order_hue
                        else:
                            global_hue_categories = [None]

                if only_numeric:
                    # ➔ Plot numeric columns one-by-one (KDEs)
                    selected_columns = st.sidebar.multiselect(
                        "Select Numeric Columns for Ridge Plot",
                        options=plot_data.select_dtypes(include=['number']).columns.tolist(),
                        default=plot_data.select_dtypes(include=['number']).columns.tolist()
                    )

                    n_rows = len(selected_columns)

                    if n_rows == 0:
                        st.warning("Please select at least one numeric variable.")
                        st.stop()

                    fig = make_subplots(
                        rows=n_rows,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.04,
                        row_titles=[str(c) for c in selected_columns]
                    )

                    for idx, col in enumerate(selected_columns):
                        x_data = plot_data[col].dropna()
                        if len(x_data) < 2:
                            continue

                        kde = gaussian_kde(x_data)
                        x_values = np.linspace(x_data.min(), x_data.max(), 200)
                        density = kde(x_values)

                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=density / density.max(),
                                mode='lines',
                                fill='tozeroy',
                                name=col,
                                line=dict(color=PALETTE[idx % len(PALETTE)]),
                                showlegend=False
                            ),
                            row=idx+1,
                            col=1
                        )

                        fig.update_yaxes(title="Density", row=idx+1, col=1)

                    fig.update_layout(
                        title="Ridge Plot of Selected Numeric Variables",
                        width=900,
                        height=max(300, n_rows * height),
                        showlegend=False
                    )

                    fig.update_xaxes(title="Value", row=n_rows, col=1)

                else:
                    # ➔ Normal faceted ridges by var_y (categorical, few categories, sorted manually)
                    fig = make_subplots(
                        rows=n_rows,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.04,
                        row_titles=[str(y) for y in y_values_unique]
                    )

                    x_range = [plot_data[var_x].dropna().min(), plot_data[var_x].dropna().max()]
                    x_values = np.linspace(x_range[0], x_range[1], 200)

                    for idx, y_val in enumerate(y_values_unique):
                        row_idx = idx + 1
                        subset = plot_data[plot_data[var_y] == y_val]

                        if subset.empty:
                            continue

                        if hue_param != var_y and hue_param in subset.columns:
                            hue_values = subset[hue_param].dropna().unique()
                        else:
                            hue_values = [None]

                        for j, hue_val in enumerate(hue_values):
                            hue_subset = subset[subset[hue_param] == hue_val] if hue_val is not None else subset
                            x_data = hue_subset[var_x].dropna()
                            if len(x_data) < 2:
                                continue

                            kde = gaussian_kde(x_data)
                            density = kde(x_values)

                            if hue_val in global_hue_categories:
                                color_idx = global_hue_categories.index(hue_val)
                            else:
                                color_idx = 0

                            fig.add_trace(
                                go.Scatter(
                                    x=x_values,
                                    y=density / density.max(),
                                    mode='lines',
                                    fill='tozeroy',
                                    name=f"{hue_val}" if hue_val is not None else str(y_val),
                                    line=dict(color=PALETTE[color_idx % len(PALETTE)]),
                                    showlegend=(row_idx == 1)
                                ),
                                row=row_idx,
                                col=1
                            )

                        fig.update_yaxes(title="Density", row=row_idx, col=1)

                    fig.update_layout(
                        title=f"Ridge Plot Faceted by {var_y}",
                        width=900,
                        height=max(300, n_rows * height),
                        showlegend=True
                    )

                    fig.update_xaxes(title=var_x, row=n_rows, col=1)

                st.plotly_chart(fig, use_container_width=True)

            # --- HISTOGRAM PLOT ---
            elif plot_type == 'histogram':
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                import pandas as pd
                from plotly.subplots import make_subplots

                # Variables depending on risk_it_all
                x_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                facet_cols = ['None'] + (categorical_cols if not risk_it_all else data.columns.tolist())

                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                hue_var = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0)
                facet_col = st.sidebar.selectbox("Facet Column (optional)", facet_cols, index=0)
                facet_row = st.sidebar.selectbox("Facet Row (optional)", facet_cols, index=0)

                multiple = st.sidebar.selectbox("Multiple", ["layer", "dodge", "stack"])
                stat = st.sidebar.selectbox("Stat", ["count", "probability", "percent", "density"])
                element = st.sidebar.selectbox("Element", ["bars", "step"])
                common_norm = st.sidebar.checkbox("Common Norm", value=False)
                cumulative = st.sidebar.checkbox("Cumulative", value=False)

                barmode = 'overlay' if multiple == 'layer' else multiple

                # Copy data to work on
                plot_data = data.copy()

                # Force hue_var to string if needed
                if hue_var != "None" and hue_var in plot_data.columns:
                    plot_data[hue_var] = plot_data[hue_var].astype(str)

                # --- Custom Ordering Section ---
                category_orders = {}

                # X Variable Custom Order
                if var_x in plot_data.columns and plot_data[var_x].dtype.name in ['object', 'category']:
                    custom_order_x = st.sidebar.multiselect(
                        f"Custom Order for {var_x}",
                        options=plot_data[var_x].dropna().unique().tolist(),
                        default=sorted(plot_data[var_x].dropna().unique().tolist()),
                        key="order_x_hist"
                    )
                    plot_data[var_x] = pd.Categorical(plot_data[var_x], categories=custom_order_x, ordered=True)
                    category_orders[var_x] = custom_order_x

                # Hue Variable Custom Order
                if hue_var != 'None' and hue_var in plot_data.columns and plot_data[hue_var].dtype.name in ['object', 'category']:
                    custom_order_hue = st.sidebar.multiselect(
                        f"Custom Order for hue {hue_var}",
                        options=plot_data[hue_var].dropna().unique().tolist(),
                        default=sorted(plot_data[hue_var].dropna().unique().tolist()),
                        key="order_hue_hist"
                    )
                    plot_data[hue_var] = pd.Categorical(plot_data[hue_var], categories=custom_order_hue, ordered=True)
                    category_orders[hue_var] = custom_order_hue

                # Facet Column Custom Order
                if facet_col != 'None' and facet_col in plot_data.columns and plot_data[facet_col].dtype.name in ['object', 'category']:
                    custom_order_facet_col = st.sidebar.multiselect(
                        f"Custom Order for {facet_col}",
                        options=plot_data[facet_col].dropna().unique().tolist(),
                        default=sorted(plot_data[facet_col].dropna().unique().tolist()),
                        key="order_facet_col_hist"
                    )
                    plot_data[facet_col] = pd.Categorical(plot_data[facet_col], categories=custom_order_facet_col, ordered=True)
                    category_orders[facet_col] = custom_order_facet_col

                # Facet Row Custom Order
                if facet_row != 'None' and facet_row in plot_data.columns and plot_data[facet_row].dtype.name in ['object', 'category']:
                    custom_order_facet_row = st.sidebar.multiselect(
                        f"Custom Order for {facet_row}",
                        options=plot_data[facet_row].dropna().unique().tolist(),
                        default=sorted(plot_data[facet_row].dropna().unique().tolist()),
                        key="order_facet_row_hist"
                    )
                    plot_data[facet_row] = pd.Categorical(plot_data[facet_row], categories=custom_order_facet_row, ordered=True)
                    category_orders[facet_row] = custom_order_facet_row

                # --- Plotting ---
                if element == "step":
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    for hue_val in plot_data[hue_var].dropna().unique() if hue_var != "None" else [None]:
                        subset = plot_data[plot_data[hue_var] == hue_val] if hue_var != "None" else plot_data
                        fig.add_trace(
                            go.Histogram(
                                x=subset[var_x],
                                histnorm=stat if stat != "count" else None,
                                cumulative_enabled=cumulative,
                                opacity=0.75 if multiple == 'layer' else 1.0,
                                name=str(hue_val) if hue_val is not None else var_x,
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
                    fig = px.histogram(
                        plot_data,
                        x=var_x,
                        color=hue_var if hue_var != "None" else None,
                        facet_col=facet_col if facet_col != "None" else None,
                        facet_row=facet_row if facet_row != "None" else None,
                        barmode=barmode,
                        histnorm=stat if stat != "count" else None,
                        cumulative=cumulative,
                        category_orders=category_orders,
                        color_discrete_sequence=PALETTE,
                        width=800,
                        height=600
                    )

                st.plotly_chart(fig, use_container_width=True)

            # --- DENSITY 1 or 2 (Unified Logic) ---
            elif plot_type in ["density 1", "density 2"]:
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                from scipy.stats import gaussian_kde
                import pandas as pd

                x_cols = data.columns if risk_it_all else numeric_cols
                cat_cols = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

                plot_data = data.copy()

                # --- Sidebar Inputs ---
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0) #, key="density_var_x")

                if plot_type == "density 2":
                    y_cols = data.columns if risk_it_all else numeric_cols
                    var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0) #, key="density_var_y")
                else:
                    var_y = None

                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else cat_cols)
                hue_var = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0) #, key="density_hue_var")

                facet_cols = ['None'] + (data.columns.tolist() if risk_it_all else cat_cols)
                facet_col = st.sidebar.selectbox("Facet Column (optional)", facet_cols, index=0) #, key="density_facet_col")
                facet_row = st.sidebar.selectbox("Facet Row (optional)", facet_cols, index=0) #, key="density_facet_row")

                if plot_type == "density 1":
                    multiple = st.sidebar.selectbox("Multiple", ["layer", "stack"], key="density1_multiple")
                    common_norm = st.sidebar.checkbox("Common Norm (Normalize)", value=False, key="density1_common_norm")
                    cumulative = st.sidebar.checkbox("Cumulative Density", value=False, key="density1_cumulative")
                else:
                    kind = st.sidebar.selectbox("Kind", ["hist", "kde"], key="density2_kind")
                    common_norm = st.sidebar.checkbox("Common Norm (Normalize)", value=False, key="density2_common_norm")
                    rug = st.sidebar.checkbox("Add Rug Plot?", value=False, key="density2_rug")

                # --- Category Orders ---
                category_orders = {}

                def add_category_order(var_name, prefix):
                    if var_name and var_name != 'None' and plot_data[var_name].dtype.name in ['object', 'category']:
                        custom_order = st.sidebar.multiselect(
                            f"Custom Order for {prefix} {var_name}",
                            options=plot_data[var_name].dropna().unique().tolist(),
                            default=sorted(plot_data[var_name].dropna().unique().tolist()),
                            key=f"{prefix}_custom_order_{var_name}"
                        )
                        plot_data[var_name] = pd.Categorical(plot_data[var_name], categories=custom_order, ordered=True)
                        category_orders[var_name] = custom_order

                add_category_order(var_x, 'x')
                if var_y: add_category_order(var_y, 'y')
                add_category_order(hue_var, 'hue')
                add_category_order(facet_col, 'col')
                add_category_order(facet_row, 'row')

                # --- Now Build the Plots ---

                ## --- DENSITY 1 ---
                if plot_type == "density 1":
                    facet_active = (facet_col != 'None') or (facet_row != 'None')

                    if facet_active:
                        facets = [col for col in [facet_row, facet_col] if col != 'None']
                        if facets:
                            # Filter plot_data to only include rows with user-selected categories and no NaN in facet columns
                            filtered_plot_data = plot_data.copy()
                            for facet in facets:
                                if facet in category_orders and category_orders[facet]:  # Ensure category_orders[facet] is not empty
                                    filtered_plot_data = filtered_plot_data[
                                        filtered_plot_data[facet].isin(category_orders[facet])
                                    ]
                                # Remove rows with NaN in the facet column
                                filtered_plot_data = filtered_plot_data.dropna(subset=[facet])

                            # Create facet_combinations from the filtered data
                            if not filtered_plot_data.empty:
                                facet_combinations = filtered_plot_data[facets].drop_duplicates()
                                for facet in facets:
                                    if facet in category_orders:
                                        facet_combinations[facet] = pd.Categorical(
                                            facet_combinations[facet],
                                            categories=category_orders[facet],
                                            ordered=True
                                        )
                                # Sort facet_combinations by facets to respect custom order
                                facet_combinations = facet_combinations.sort_values(by=facets)
                                n_rows = facet_combinations[facet_row].nunique() if facet_row != 'None' else 1
                                n_cols = facet_combinations[facet_col].nunique() if facet_col != 'None' else 1
                            else:
                                # If no data remains after filtering, treat as non-faceted
                                facet_combinations = pd.DataFrame([{}])
                                n_rows, n_cols = 1, 1
                                facet_active = False  # Disable faceting if no valid data
                        else:
                            facet_combinations = pd.DataFrame([{}])
                            n_rows, n_cols = 1, 1
                            facet_active = False
                    else:
                        n_rows, n_cols = 1, 1

                    subplot_titles = []
                    if facet_active and facets:
                        for _, row in facet_combinations.iterrows():
                            title = []
                            if facet_row != 'None':
                                title.append(f"{facet_row}: {row[facet_row]}")
                            if facet_col != 'None':
                                title.append(f"{facet_col}: {row[facet_col]}")
                            subplot_titles.append(" | ".join(title))

                    # Only create subplots for the number of valid facet combinations
                    if facet_active and subplot_titles:
                        fig = make_subplots(
                            rows=n_rows,
                            cols=n_cols,
                            subplot_titles=subplot_titles,
                            horizontal_spacing=0.08,
                            vertical_spacing=0.1,
                            shared_xaxes=True,
                        )
                    else:
                        # If no facets or no valid combinations, create a single subplot
                        fig = make_subplots(
                            rows=1,
                            cols=1,
                            horizontal_spacing=0.08,
                            vertical_spacing=0.1,
                            shared_xaxes=True,
                        )
                        n_rows, n_cols = 1, 1

                    x_range = np.linspace(plot_data[var_x].dropna().min(), plot_data[var_x].dropna().max(), 200)

                    # Define facet_groups based on sorted facet_combinations
                    if facet_active and facets and not facet_combinations.empty and not facet_combinations.equals(pd.DataFrame([{}])):
                        facet_groups = [(tuple(row[facets]), filtered_plot_data[
                            (filtered_plot_data[facets].eq(row[facets]).all(axis=1)) if len(facets) > 1 else (filtered_plot_data[facets[0]] == row[facets[0]])
                        ]) for _, row in facet_combinations.iterrows()]
                    else:
                        facet_groups = [((), plot_data)]

                    for idx, (facet_vals, group_df) in enumerate(facet_groups):
                        if facet_active and facets:
                            if group_df.empty:
                                continue  # Skip empty groups to avoid plotting empty panels
                            facet_vals = list(facet_vals) if isinstance(facet_vals, tuple) else [facet_vals]
                            # Safely compute row_idx and col_idx, ensuring facet_vals does not contain NaN
                            row_idx = list(facet_combinations[facet_row].unique()).index(facet_vals[0]) + 1 if facet_row != 'None' else 1
                            col_idx = list(facet_combinations[facet_col].unique()).index(facet_vals[-1]) + 1 if facet_col != 'None' else 1
                        else:
                            row_idx, col_idx = 1, 1

                        hue_values = group_df[hue_var].dropna().unique() if hue_var != 'None' else [None]
                        # Sort hue_values according to category_orders[hue_var] if it exists
                        if hue_var != 'None' and hue_var in category_orders:
                            # Only include hue values that are present in the current group_df
                            hue_values = [v for v in category_orders[hue_var] if v in hue_values]

                        offset = 0
                        for j, hue_val in enumerate(hue_values):
                            subset = group_df[group_df[hue_var] == hue_val] if hue_var != 'None' else group_df
                            x_data = subset[var_x].dropna()
                            if len(x_data) < 2:
                                continue
                            try:
                                kde = gaussian_kde(x_data)
                                density = kde(x_range)
                                if cumulative:
                                    density = np.cumsum(density)
                                    density = density / density[-1]
                                if common_norm and not cumulative:
                                    density = density / density.max()
                                y_values = density + offset if multiple == "stack" else density

                                fig.add_trace(
                                    go.Scatter(
                                        x=x_range,
                                        y=y_values,
                                        mode='lines',
                                        fill='tozeroy' if multiple == "stack" else None,
                                        name=f"{hue_val}" if hue_var != 'None' else str(var_x),
                                        line=dict(color=PALETTE[j % len(PALETTE)]),
                                        showlegend=(idx == 0),
                                    ),
                                    row=row_idx,
                                    col=col_idx
                                )

                                if multiple == "stack":
                                    offset += y_values.max()
                            except Exception:
                                continue

                    fig.update_layout(
                        title=f"Density Plot of {var_x}",
                        width=900,
                        height=300 * max(n_rows, 1),
                        showlegend=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                ## --- DENSITY 2 ---
                else:
                    if kind == "hist":
                        fig = px.density_heatmap(
                            plot_data,
                            x=var_x,
                            y=var_y,
                            facet_col=facet_col if facet_col != 'None' else None,
                            facet_row=facet_row if facet_row != 'None' else None,
                            color_continuous_scale='Viridis',
                            histnorm='probability density' if not common_norm else None,
                            marginal_x='rug' if rug else None,
                            marginal_y='rug' if rug else None,
                            width=800,
                            height=600,
                            category_orders=category_orders
                        )
                    else:
                        fig = px.density_contour(
                            plot_data,
                            x=var_x,
                            y=var_y,
                            color=hue_var if hue_var != 'None' else None,
                            facet_col=facet_col if facet_col != 'None' else None,
                            facet_row=facet_row if facet_row != 'None' else None,
                            color_discrete_sequence=PALETTE,
                            marginal_x='rug' if rug else None,
                            marginal_y='rug' if rug else None,
                            width=800,
                            height=600,
                            category_orders=category_orders
                        )

                    st.plotly_chart(fig, use_container_width=True)

            # Scatter
            elif plot_type == 'scatter':
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import pandas as pd

                # Define available columns depending on risk_it_all option
                x_cols = data.columns if risk_it_all else numeric_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                style_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                size_cols = ['None'] + (data.columns.tolist() if risk_it_all else numeric_cols)
                facet_cols = ['None'] + (categorical_cols if not risk_it_all else data.columns.tolist())

                # Sidebar selections
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0)
                style = st.sidebar.selectbox("Style (Symbol)", style_cols, index=0)
                size = st.sidebar.selectbox("Size (Bubble Size)", size_cols, index=0)
                facet_col = st.sidebar.selectbox("Facet Column (optional)", facet_cols, index=0)
                facet_row = st.sidebar.selectbox("Facet Row (optional)", facet_cols, index=0)

                alpha = st.sidebar.slider("Alpha (Opacity)", 0.0, 1.0, 0.8, 0.01)
                size_max = st.sidebar.slider("Max Marker Size", 5, 100, 10, 5)

                enhance_size = st.sidebar.selectbox(
                    "Enhance Size Differences",
                    options=["None", "Min-Max Normalize"],
                    index=0
                )

                plot_data = data.copy()

                # Custom Ordering for Categorical Variables
                category_orders = {}

                # Helper function to add category ordering
                def add_category_order(var_name, label):
                    if var_name != 'None' and var_name in plot_data.columns and plot_data[var_name].dtype.name in ['object', 'category']:
                        custom_order = st.sidebar.multiselect(
                            f"Custom Order for {label} ({var_name})",
                            options=plot_data[var_name].dropna().unique().tolist(),
                            default=sorted(plot_data[var_name].dropna().unique().tolist()),
                            help=f"Select the order of categories for {var_name}."
                        )
                        if custom_order:  # Only apply if the user has made a selection
                            plot_data[var_name] = pd.Categorical(plot_data[var_name], categories=custom_order, ordered=True)
                            category_orders[var_name] = custom_order

                # Add sorting for hue
                add_category_order(hue, "Hue")

                # Add sorting for facet_col
                add_category_order(facet_col, "Facet Column")

                # Add sorting for facet_row
                add_category_order(facet_row, "Facet Row")

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
                    category_orders=category_orders  # Add category_orders to respect user-specified sorting
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
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import pandas as pd

                x_cols = data.columns if risk_it_all else categorical_cols
                y_cols = data.columns if risk_it_all else categorical_cols
                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                col_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                row_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                size_cols = ['None'] + (data.columns.tolist() if risk_it_all else numeric_cols)

                # Controls
                st.sidebar.header("Catplot Options")
                var_x = st.sidebar.selectbox(
                    "X Variable",
                    x_cols,
                    index=0,
                    help="Choose a categorical variable for the x-axis."
                )
                var_y = st.sidebar.selectbox(
                    "Y Variable",
                    y_cols,
                    index=0,
                    #help="Choose a categorical variable for the y-axis."
                )
                hue = st.sidebar.selectbox(
                    "Hue (Color)",
                    hue_cols,
                    index=0,
                    #help="Choose a categorical variable for color grouping (optional)."
                )
                col = st.sidebar.selectbox(
                    "Facet Column (optional)",
                    col_cols,
                    index=0,
                    #help="Choose a categorical variable to facet the plot into columns (optional; selecting a variable enables faceting)."
                )
                row = st.sidebar.selectbox(
                    "Facet Row (optional)",
                    row_cols,
                    index=0,
                    #help="Choose a categorical variable to facet the plot into rows (optional; selecting a variable enables faceting)."
                )
                size_var = st.sidebar.selectbox(
                    "Size Variable",
                    size_cols,
                    index=0,
                    help="Choose a numerical variable to control marker sizes (optional)."
                )

                kind = st.sidebar.selectbox(
                    "Kind",
                    ["strip", "swarm"],
                    help="Choose the type of catplot: strip or swarm."
                )

                # Global point size if no size variable selected
                global_point_size = st.sidebar.slider(
                    "Global Marker Size",
                    5,
                    50,
                    10,
                    5,
                    help="Set the marker size when no Size Variable is selected."
                )

                plot_data = data.copy()

                # Custom Ordering for Categorical Variables
                category_orders = {}

                # Order for X Variable
                if var_x in plot_data.columns and plot_data[var_x].dtype.name in ['object', 'category']:
                    custom_order_x = st.sidebar.multiselect(
                        f"Custom Order for X Variable ({var_x})",
                        options=plot_data[var_x].dropna().unique().tolist(),
                        default=sorted(plot_data[var_x].dropna().unique().tolist()),
                        help=f"Select the order of categories for {var_x} to display on the x-axis."
                    )
                    if custom_order_x:
                        plot_data[var_x] = pd.Categorical(plot_data[var_x], categories=custom_order_x, ordered=True)
                        category_orders[var_x] = custom_order_x

                # Order for Y Variable
                if var_y in plot_data.columns and plot_data[var_y].dtype.name in ['object', 'category']:
                    custom_order_y = st.sidebar.multiselect(
                        f"Custom Order for Y Variable ({var_y})",
                        options=plot_data[var_y].dropna().unique().tolist(),
                        default=sorted(plot_data[var_y].dropna().unique().tolist()),
                        help=f"Select the order of categories for {var_y} to display on the y-axis."
                    )
                    if custom_order_y:
                        plot_data[var_y] = pd.Categorical(plot_data[var_y], categories=custom_order_y, ordered=True)
                        category_orders[var_y] = custom_order_y

                # Order for Hue
                if hue != 'None' and hue in plot_data.columns and plot_data[hue].dtype.name in ['object', 'category']:
                    custom_order_hue = st.sidebar.multiselect(
                        f"Custom Order for Hue ({hue})",
                        options=plot_data[hue].dropna().unique().tolist(),
                        default=sorted(plot_data[hue].dropna().unique().tolist()),
                        help=f"Select the order of categories for {hue} to display in the color grouping."
                    )
                    if custom_order_hue:
                        plot_data[hue] = pd.Categorical(plot_data[hue], categories=custom_order_hue, ordered=True)
                        category_orders[hue] = custom_order_hue

                # Order for Facet Column
                if col != 'None' and col in plot_data.columns and plot_data[col].dtype.name in ['object', 'category']:
                    custom_order_col = st.sidebar.multiselect(
                        f"Custom Order for Facet Column ({col})",
                        options=plot_data[col].dropna().unique().tolist(),
                        default=sorted(plot_data[col].dropna().unique().tolist()),
                        help=f"Select the order of categories for {col} to display in the facet columns."
                    )
                    if custom_order_col:
                        plot_data[col] = pd.Categorical(plot_data[col], categories=custom_order_col, ordered=True)
                        category_orders[col] = custom_order_col

                # Order for Facet Row
                if row != 'None' and row in plot_data.columns and plot_data[row].dtype.name in ['object', 'category']:
                    custom_order_row = st.sidebar.multiselect(
                        f"Custom Order for Facet Row ({row})",
                        options=plot_data[row].dropna().unique().tolist(),
                        default=sorted(plot_data[row].dropna().unique().tolist()),
                        help=f"Select the order of categories for {row} to display in the facet rows."
                    )
                    if custom_order_row:
                        plot_data[row] = pd.Categorical(plot_data[row], categories=custom_order_row, ordered=True)
                        category_orders[row] = custom_order_row

                # Prepare the plot
                if col != 'None' or row != 'None':
                    # Faceting enabled: Use make_subplots for manual faceting
                    col_vals = plot_data[col].dropna().unique() if col != 'None' else [None]
                    row_vals = plot_data[row].dropna().unique() if row != 'None' else [None]

                    n_cols = max(len(col_vals), 1)
                    n_rows = max(len(row_vals), 1)

                    # Create subplot figure
                    fig = make_subplots(
                        rows=n_rows,
                        cols=n_cols,
                        subplot_titles=[
                            f"{row}: {r} | {col}: {c}" if row != 'None' and col != 'None' else
                            f"{col}: {c}" if col != 'None' else
                            f"{row}: {r}" if row != 'None' else ""
                            for r in row_vals for c in col_vals
                        ],
                        shared_yaxes=True,
                        shared_xaxes=True,
                        horizontal_spacing=0.1,
                        vertical_spacing=0.12
                    )

                    # Color mapping for hue
                    if hue != 'None':
                        hue_values = plot_data[hue].dropna().unique()
                    else:
                        hue_values = [None]

                    color_map = {}
                    for idx, hv in enumerate(hue_values):
                        color_map[hv] = PALETTE[idx % len(PALETTE)]

                    # Add traces for each facet
                    for i_row, row_val in enumerate(row_vals):
                        for i_col, col_val in enumerate(col_vals):
                            row_idx = i_row + 1
                            col_idx = i_col + 1

                            # Filter data by facet values
                            sub_data = plot_data.copy()
                            if row != 'None':
                                sub_data = sub_data[sub_data[row] == row_val]
                            if col != 'None':
                                sub_data = sub_data[sub_data[col] == col_val]

                            if sub_data.empty:
                                continue

                            for hue_val in hue_values:
                                # Filter subset by hue
                                subset = sub_data.copy()
                                if hue_val is not None:
                                    subset = subset[subset[hue] == hue_val]

                                if subset.empty:
                                    continue

                                # Get color for this hue
                                marker_color = color_map[hue_val]

                                # Create scatter trace
                                scatter = go.Scatter(
                                    x=subset[var_x],
                                    y=subset[var_y],
                                    mode='markers',
                                    name=str(hue_val) if hue_val else "Points",
                                    marker=dict(
                                        color=marker_color,
                                        size=global_point_size if size_var == 'None' or not pd.api.types.is_numeric_dtype(plot_data[size_var]) else subset[size_var],
                                        sizemode='diameter',
                                        sizeref=2.*subset[size_var].max()/(15.**2) if size_var != 'None' and pd.api.types.is_numeric_dtype(plot_data[size_var]) else None,
                                        sizemin=4 if size_var != 'None' and pd.api.types.is_numeric_dtype(plot_data[size_var]) else None
                                    ),
                                    showlegend=(row_idx == 1 and col_idx == 1),
                                    legendgroup=str(hue_val) if hue_val else None
                                )

                                if kind == "swarm":
                                    scatter.update(marker=dict(sizeref=2.*subset[size_var].max()/(15.**2) if size_var != 'None' else None))
                                    # Simulate swarm by adding jitter
                                    scatter.update(x=jittered_values(subset[var_x], 0.3))

                                fig.add_trace(scatter, row=row_idx, col=col_idx)

                    # Update layout
                    fig.update_layout(
                        width=800,
                        height=400 * n_rows,  # Scale height based on number of rows
                        title="Catplot with Faceting",
                        showlegend=True,
                        legend_title=hue if hue != 'None' else None
                    )

                else:
                    # No faceting: Use px.strip
                    fig = px.strip(
                        plot_data,
                        x=var_x,
                        y=var_y,
                        color=hue if hue != 'None' else None,
                        color_discrete_sequence=PALETTE,
                        category_orders=category_orders,
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
                import plotly.express as px
                import plotly.graph_objects as go
                import statsmodels.api as sm
                import pandas as pd

                x_cols = data.columns if risk_it_all else numeric_cols
                y_cols = data.columns if risk_it_all else numeric_cols
                hue_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)

                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue (Color)", hue_cols, index=0)
                order = st.sidebar.selectbox("Order of Polynomial Fit", [1, 2, 3], index=0)
                ci_level = st.sidebar.selectbox("Confidence Interval (%)", [68, 90, 95, 99], index=2)
                use_hue = st.sidebar.checkbox("Use Hue?", value=True)

                plot_data = data.copy()
                fig = go.Figure()

                # Safely define groups
                if use_hue and hue != 'None' and hue in plot_data.columns:
                    groups = plot_data[hue].dropna().unique()
                else:
                    groups = [None]

                for idx, g in enumerate(groups):
                    if g is not None and hue in plot_data.columns:
                        subset = plot_data[plot_data[hue] == g]
                    else:
                        subset = plot_data

                    X = subset[var_x]
                    Y = subset[var_y]

                    # Drop NaNs
                    mask = (~X.isna()) & (~Y.isna())
                    X = X[mask]
                    Y = Y[mask]

                    if X.empty or Y.empty:
                        continue

                    # Scatter plot
                    fig.add_trace(go.Scatter(
                        x=X,
                        y=Y,
                        mode='markers',
                        marker=dict(color=PALETTE[idx % len(PALETTE)], size=5),
                        name=str(g) if g is not None else "Data",
                        showlegend=True
                    ))

                    # Polynomial regression
                    X_design = sm.add_constant(np.vander(X, N=order+1, increasing=True))
                    model = sm.OLS(Y, X_design).fit()

                    # Prediction
                    x_pred = np.linspace(X.min(), X.max(), 100)
                    X_pred_design = sm.add_constant(np.vander(x_pred, N=order+1, increasing=True))
                    y_pred = model.predict(X_pred_design)

                    # Confidence intervals
                    pred_summary = model.get_prediction(X_pred_design).summary_frame(alpha=(1 - ci_level / 100))

                    # Regression line
                    fig.add_trace(go.Scatter(
                        x=x_pred,
                        y=y_pred,
                        mode='lines',
                        line=dict(color=PALETTE[idx % len(PALETTE)]),
                        name=f"Fit: {g}" if g is not None else "Fit",
                        showlegend=True
                    ))

                    # Confidence band
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x_pred, x_pred[::-1]]),
                        y=np.concatenate([pred_summary['mean_ci_upper'], pred_summary['mean_ci_lower'][::-1]]),
                        fill='toself',
                        fillcolor=f"rgba(0,100,80,0.2)",
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

            # Heatmap Numerical Section
            if plot_type == 'heatmap':
                import plotly.express as px
                import pandas as pd
                import numpy as np

                # Setup columns
                all_cols = data.columns.tolist()
                cat_cols = [None] + [col for col in all_cols if not pd.api.types.is_numeric_dtype(data[col]) or isinstance(data[col].dtype, pd.CategoricalDtype)]
                num_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(data[col])]

                # Controls
                st.sidebar.header("Heatmap Numerical Options")
                cat_var = st.sidebar.selectbox(
                    "Select Categorical Variable (Optional)",
                    cat_cols,
                    index=0,
                    help="Choose a categorical variable to use as rows in the heatmap. Leave as None to plot all numerical columns as rows."
                )

                num_var = None
                if cat_var is not None:
                    if not num_cols:
                        st.warning("No numerical columns available to select as the numerical variable.")
                        return
                    default_num = num_cols[0]
                    num_var = st.sidebar.selectbox(
                        "Select Numerical Variable (Required)",
                        num_cols,
                        index=num_cols.index(default_num),
                        help="Choose a numerical variable to use as values in the heatmap."
                    )

                colormap = st.sidebar.selectbox(
                    "Select Colormap",
                    ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Reds", "Greens"],
                    index=0,
                    help="Choose a colormap for the heatmap visualization."
                )

                plot_data = data.copy()

                # Custom Ordering for Categorical Variable (Added)
                if cat_var is not None and cat_var in plot_data.columns:
                    custom_order = st.sidebar.multiselect(
                        f"Custom Order for {cat_var}",
                        options=plot_data[cat_var].dropna().unique().tolist(),
                        default=sorted(plot_data[cat_var].dropna().unique().tolist()),
                        help=f"Select the order of categories for {cat_var} to display in the heatmap."
                    )
                    if custom_order:  # Only apply custom order if the user has made a selection
                        plot_data[cat_var] = pd.Categorical(plot_data[cat_var], categories=custom_order, ordered=True)

                # Debug: Inspect plot_data shape and columns
                #st.write(f"Debug: plot_data shape: {plot_data.shape}")
                #st.write(f"Debug: plot_data columns: {plot_data.columns.tolist()}")

                # Prepare heatmap data
                if cat_var is None:
                    # Case 1: All columns numerical
                    # Check if all columns are numerical
                    if not all(pd.api.types.is_numeric_dtype(plot_data[col]) for col in plot_data.columns):
                        st.warning("All columns must be numerical for Heatmap Numerical mode without a categorical variable.")
                        return

                    # Select only numerical columns (ensures consistency)
                    numerical_data = plot_data.select_dtypes(include=np.number)

                    # Debug: Inspect numerical_data shape and columns
                    #st.write(f"Debug: numerical_data shape: {numerical_data.shape}")
                    #st.write(f"Debug: numerical_data columns: {numerical_data.columns.tolist()}")

                    # Transpose the DataFrame: columns become rows, rows become columns
                    heatmap_data = numerical_data.transpose()
                    x_labels = [f"data{i}" for i in range(len(numerical_data))]
                    y_labels = numerical_data.columns.tolist()
                    title = "Heatmap Numerical: Observations vs Variables"

                    # Debug: Inspect heatmap_data shape
                    #st.write(f"Debug: heatmap_data shape after transpose: {heatmap_data.shape}")

                    # Calculate dynamic height based on number of rows (columns in original data)
                    num_rows = len(y_labels)  # Number of numerical columns (rows in heatmap)
                    base_height = 600
                    height = max(base_height, num_rows * 20)  # Scale height: at least 20 pixels per row

                else:
                    # Case 2: Categorical variable for rows, numerical variable for columns
                    if num_var is None:
                        st.warning("Please select a numerical variable for Heatmap Numerical mode.")
                        return

                    if not pd.api.types.is_numeric_dtype(plot_data[num_var]):
                        st.warning(f"Selected numerical variable '{num_var}' must be numerical.")
                        return

                    # Check if cat_var is categorical or object
                    is_categorical = False
                    if isinstance(plot_data[cat_var].dtype, pd.CategoricalDtype):
                        is_categorical = True
                    elif plot_data[cat_var].dtype.name == 'object':
                        is_categorical = True

                    if not is_categorical:
                        st.warning(f"Selected categorical variable '{cat_var}' must be categorical or object type.")
                        return

                    # Add a temporary observation ID column to use as pivot columns
                    plot_data['obs_id'] = [f"Obs {i}" for i in range(len(plot_data))]

                    # Pivot the data: cat_var as rows, obs_id as columns, num_var as values
                    heatmap_data = plot_data.pivot(columns='obs_id', values=num_var, index=cat_var)
                    x_labels = heatmap_data.columns.tolist()  # Observation IDs
                    y_labels = heatmap_data.index.tolist()   # Unique values of cat_var, ordered by custom_order if set
                    title = f"Heatmap Numerical: {cat_var} vs Observations"

                    # Debug: Inspect heatmap_data shape
                    #st.write(f"Debug: heatmap_data shape after pivot: {heatmap_data.shape}")

                    # Calculate dynamic height based on number of rows (unique cat_var values)
                    num_rows = len(y_labels)  # Number of unique categorical values (rows in heatmap)
                    base_height = 600
                    height = max(base_height, num_rows * 20)  # Scale height: at least 20 pixels per row

                # Plot the heatmap
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Observations", y="Variables" if cat_var is None else cat_var, color="Value"),
                    x=x_labels,
                    y=y_labels,
                    color_continuous_scale=colormap,
                    title=title,
                    width=800,
                    height=height,
                    aspect='auto'  # Prevent Plotly from enforcing a square aspect ratio
                )

                # Update layout to ensure full visibility
                fig.update_layout(
                    margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins for better visibility
                    yaxis=dict(tickfont=dict(size=10)),    # Adjust font size for y-axis labels
                )

                st.plotly_chart(fig, use_container_width=True)

    # Render the plot if data is available
    if not data.empty:
        render_plot()
        st.markdown(
            """
            <hr style="margin-top:2em;margin-bottom:1em;">
            <div style="text-align: center; color: gray; font-size: 0.9em;">
            Explore. Understand. Inspire. 🚀 | Powered by Streamlit + Plotly.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("The dataset is empty after preprocessing. Please check your data or selections in the Home section.")

# Helper function to simulate jitter for swarm plot
def jittered_values(series, jitter_amount=0.3):
    # Add random jitter to categorical values for swarm effect
    if pd.api.types.is_numeric_dtype(series):
        return series + np.random.uniform(-jitter_amount, jitter_amount, size=len(series))
    else:
        # For categorical data, convert to numeric codes and add jitter
        codes = pd.Categorical(series).codes
        return codes + np.random.uniform(-jitter_amount, jitter_amount, size=len(series))