import streamlit as st
import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.cluster import hierarchy
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Custom CSS to make the plot region wider
st.set_page_config(layout="wide")

def plot_data():
    """Interactive plotting function for exploring a user-uploaded CSV dataset in Streamlit."""
    
    # Define plot types
    PLOT_TYPES = [
        'bars', 'boxes', 'ridges', 'histogram', 'density 1', 'density 2', 
        'scatter', 'catplot', 'missingno', 'correlation', 'clustermap', 
        'pairplot', 'regression'
    ]

    # Streamlit UI for file upload
    st.sidebar.header("Upload and Plot Selection")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is None:
        st.write("Please upload a CSV file to begin.")
        return

    # Read and preprocess data
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return

    # Convert categorical columns to strings
    for col in data.select_dtypes(include='category').columns:
        data[col] = data[col].astype('str')

    # Streamlit UI elements for plot selection
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
            hue_var = st.sidebar.selectbox("Hue Variable", data.select_dtypes(include='object').columns, index=0)
            numeric_cols = list(data.select_dtypes(include='number').columns)
            if not numeric_cols:
                st.warning("No numeric columns available for pairplot.")
                return
            
            # Create subplot grid without shared axes
            n_cols = len(numeric_cols)
            fig = make_subplots(
                rows=n_cols, cols=n_cols,
                row_titles=numeric_cols,
                column_titles=numeric_cols,
                vertical_spacing=0.05,
                horizontal_spacing=0.05,
                shared_xaxes=False,
                shared_yaxes=False
            )
            
            # Plot KDEs on diagonal and scatters off-diagonal
            for i in range(n_cols):
                col_i = numeric_cols[i]
                for j in range(n_cols):
                    col_j = numeric_cols[j]
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
                    col_i = numeric_cols[i]
                    col_j = numeric_cols[j]
                    # Compute ranges, handling NaN values
                    x_data = data[col_j].dropna()
                    y_data = data[col_i].dropna()
                    x_range = [min(x_data), max(x_data)] if not x_data.empty else [0, 1]
                    y_range = [0, 1.2] if i == j else [min(y_data), max(y_data)] if not y_data.empty else [0, 1]
                    # Extend ranges slightly to ensure all points are visible
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
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
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
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
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
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
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
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
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
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
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
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                col_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
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
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                style_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                size_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                var_x = st.sidebar.selectbox("X Variable", x_cols, index=0)
                var_y = st.sidebar.selectbox("Y Variable", y_cols, index=0)
                hue = st.sidebar.selectbox("Hue", hue_cols, index=0)
                style = st.sidebar.selectbox("Style", style_cols, index=0)
                size = st.sidebar.selectbox("Size", size_cols, index=0)
                alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.5, 0.01)
                use_style = st.sidebar.checkbox("Use Style", value=False)
                # Handle NaN in size column
                plot_data = data
                size_param = size if size in data.columns else None
                if size_param and plot_data[size].isna().any():
                    st.warning(f"Size column '{size}' contains NaN values. Dropping rows with NaN in size.")
                    plot_data = plot_data.dropna(subset=[size])
                fig = px.scatter(plot_data, x=var_x, y=var_y, color=hue, 
                                size=size_param, 
                                symbol=style if use_style else None, 
                                opacity=alpha, 
                                color_discrete_sequence=PALETTE, 
                                width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)

            # Catplot
            elif plot_type == 'catplot':
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                col_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
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
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
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

    # Render the plot
    render_plot()

# Main app
if __name__ == "__main__":
    st.title("Interactive Data Visualization App")
    plot_data()