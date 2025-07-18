import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from plotly_calplot import calplot
import requests
import io
from streamlit_option_menu import option_menu

def time_series_analysis():
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.ensemble import IsolationForest
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    from plotly_calplot import calplot
    import pandas as pd
    import numpy as np
    import requests
    import io

    st.subheader("📈 Time Series Analysis & Anomaly Detection")

    with st.sidebar:
        analysis_section = option_menu(
            menu_title="Analysis Section",
            options=[
                "Load Data",
                "Time Series Visualization",
                "Decomposition & ACF/PACF",
                "Anomaly Detection",
                "Temporal Profiles",
                "Calendar Heatmap",
                "Heatmap Profiles",
                "Forecasting",
                "Changepoint Analysis",
                "Trend Analysis"
            ],
            icons=["upload", "bar-chart", "graph-up", "exclamation-triangle", "clock", "calendar", "heatmap", "predict", "pin-angle", "trending-up"],
            menu_icon="clock",
            default_index=0,
        )

    if analysis_section == "Load Data":
        na_values = ['', 'NA', 'N/A', 'na', 'n/a', 'NaN', 'nan', 'NULL', 'null', 'missing', '-', '--']
        df = None

        source_type = st.radio("Select data source", ["Upload CSV", "GitHub Link"])

        if source_type == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file, na_values=na_values)
                    st.success("✅ File successfully uploaded!")
                except Exception as e:
                    st.error(f"❌ Failed to load CSV: {e}")
                    st.stop()

        else:
            github_url = st.text_input("Paste GitHub CSV URL")
            if github_url:
                try:
                    github_url = github_url.strip()
                    if "github.com" in github_url and "raw.githubusercontent.com" not in github_url:
                        github_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                    if not github_url.lower().split("?")[0].endswith('.csv'):
                        st.error("❌ The provided link does not point to a CSV file. Please ensure the URL ends with '.csv'.")
                        st.stop()
                    response = requests.get(github_url, timeout=10)
                    response.raise_for_status()
                    csv_content = response.content.decode('utf-8')
                    df = pd.read_csv(io.StringIO(csv_content), na_values=na_values)
                    st.success("✅ File successfully loaded from GitHub!")
                except Exception as e:
                    st.error(f"❌ Failed to load CSV: {e}")
                    st.stop()

        if df is not None:
            date_col = st.selectbox("Select the date column", df.columns, index=0)
            date_format = st.radio(
                "Select date format",
                options=["DD/MM/YYYY (Day First)", "MM/DD/YYYY (Month First)"],
                index=0
            )
            dayfirst = date_format == "DD/MM/YYYY (Day First)"

            try:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors='coerce')
                invalid_dates = df[date_col].isna().sum()
                if invalid_dates > 0:
                    st.warning(f"⚠️ {invalid_dates} rows could not be parsed as dates and will be removed.")
                df = df.dropna(subset=[date_col])
                if df.empty:
                    st.error("❌ No valid dates remain after parsing. Please check the date column and format.")
                    st.stop()
                df = df.sort_values(by=date_col).set_index(date_col)
                st.session_state['df'] = df
                st.success(f"✅ Data loaded successfully! {len(df)} rows processed.")
                st.write("**Preview of Loaded Data (First 5 Rows):**")
                st.write(df.head())
            except Exception as e:
                st.error(f"❌ Error parsing dates: {e}")
                st.stop()
        st.stop()

    if 'df' not in st.session_state:
        st.warning("Please load data first using the 'Load Data' section.")
        return

    df = st.session_state['df']

    st.sidebar.header("Options")
    selected_col = st.sidebar.selectbox(
        "Select a variable to analyze",
        [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
    )
    df_selected = df[[selected_col]].copy()

    time_granularity = st.sidebar.selectbox(
        "Select summarization frequency",
        ["None", "Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
        index=0
    )
    aggregation_method = st.sidebar.selectbox(
        "Aggregation Method",
        ["Mean", "Sum"],
        index=0
    ) if time_granularity != "None" else "Raw"
    window = st.sidebar.number_input("Rolling Window (set 0 for no smoothing)", min_value=0, max_value=60, value=0)

    resample_map = {
        "None": None,
        "Hourly": "H",
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y"
    }
    resample_freq = resample_map[time_granularity]

    if resample_freq:
        resampled = df_selected.resample(resample_freq).agg({selected_col: aggregation_method.lower()})
        resampled.columns = ['value']
        if window > 0:
            resampled['value'] = resampled['value'].rolling(window).mean()
            resampled['std'] = resampled['value'].rolling(window).std()
        else:
            resampled['std'] = resampled['value'].std()
        df_summary = resampled
    else:
        df_summary = df_selected.copy()
        df_summary['value'] = df_summary[selected_col].rolling(window).mean() if window > 0 else df_summary[selected_col]
        df_summary['std'] = df_summary[selected_col].rolling(window).std() if window > 0 else df_summary[selected_col].std()

    if analysis_section == "Time Series Visualization":
        st.subheader("📈 Time Series Visualization")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_summary.index, y=df_summary['value'], mode='lines', name=aggregation_method if resample_freq else 'Raw'))
        if window > 0 or resample_freq:
            fig.add_trace(go.Scatter(x=df_summary.index, y=df_summary['value'] + df_summary['std'], mode='lines', name='Upper Band', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df_summary.index, y=df_summary['value'] - df_summary['std'], mode='lines', name='Lower Band', line=dict(dash='dot')))
        fig.update_layout(
            title=f"Time Series ({aggregation_method if resample_freq else 'Raw'})",
            xaxis_title="Date",
            yaxis_title=selected_col
        )
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Decomposition & ACF/PACF":
        import plotly.subplots as sp

        st.subheader("📉 Seasonal Decomposition")
        decomposition_method = st.selectbox("Select Decomposition Method", ["Statsmodels", "Prophet"], index=0)

        if decomposition_method == "Statsmodels":
            window = st.slider("Seasonal window for decomposition", 2, 60, 12)
            try:
                result = seasonal_decompose(df[selected_col], model='additive', period=window)
                fig = result.plot()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Decomposition failed: {e}. Try adjusting the window size.")
        else:
            from prophet import Prophet
            df_prophet = df_selected.reset_index()
            first_column_name = df_prophet.columns[0]
            df_prophet = df_prophet.rename(columns={first_column_name: 'ds', selected_col: 'y'})
            df_prophet = df_prophet[['ds', 'y']].copy()
            df_prophet['y'] = df_prophet['y'].fillna(method='ffill')

            if df_prophet.empty or df_prophet['ds'].isna().all() or df_prophet['y'].isna().all():
                st.error("No valid data available for Prophet decomposition. Please check the dataset.")
                return

            try:
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                model.fit(df_prophet)
                forecast = model.predict(df_prophet[['ds']])

                fig = sp.make_subplots(
                    rows=4, cols=1,
                    subplot_titles=("Trend", "Yearly Seasonality", "Weekly Seasonality", "Daily Seasonality"),
                    vertical_spacing=0.1
                )
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend'), row=1, col=1)
                if 'yearly' in forecast.columns:
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'], mode='lines', name='Yearly'), row=2, col=1)
                if 'weekly' in forecast.columns:
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['weekly'], mode='lines', name='Weekly'), row=3, col=1)
                if 'daily' in forecast.columns:
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['daily'], mode='lines', name='Daily'), row=4, col=1)

                fig.update_layout(height=800, title_text=f"Prophet Decomposition of {selected_col}", showlegend=False)
                fig.update_yaxes(title_text=selected_col, row=1, col=1)
                fig.update_yaxes(title_text="Yearly", row=2, col=1)
                fig.update_yaxes(title_text="Weekly", row=3, col=1)
                fig.update_yaxes(title_text="Daily", row=4, col=1)
                fig.update_xaxes(title_text="Date", row=4, col=1)

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prophet decomposition failed: {e}")

        st.subheader("🔁 ACF and PACF")
        fig_acf, ax = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(df[selected_col].dropna(), ax=ax[0])
        plot_pacf(df[selected_col].dropna(), ax=ax[1])
        ax[0].set_title("ACF")
        ax[1].set_title("PACF")
        st.pyplot(fig_acf)

    elif analysis_section == "Anomaly Detection":
        st.subheader("🚨 Anomaly Detection (Isolation Forest)")
        contamination = st.slider("Contamination (expected anomaly fraction)", 0.01, 0.5, 0.05, 0.01)
        df_anom = df_summary.copy().dropna(subset=['value'])
        df_anom['anomaly'] = IsolationForest(contamination=contamination, random_state=42).fit_predict(df_anom[['value']])
        df_anom['anomaly'] = df_anom['anomaly'].map({1: 0, -1: 1})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_anom.index, y=df_anom['value'], mode='lines', name=aggregation_method if resample_freq else 'Raw'))
        fig.add_trace(go.Scatter(x=df_anom[df_anom['anomaly'] == 1].index, y=df_anom[df_anom['anomaly'] == 1]['value'], mode='markers', name='Anomaly', marker=dict(color='red', size=8)))
        fig.update_layout(
            title=f"Anomaly Detection ({aggregation_method if resample_freq else 'Raw'})",
            xaxis_title="Date",
            yaxis_title=selected_col
        )
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Temporal Profiles":
        st.subheader("🕒 Temporal Profiles")
        profile_unit = st.selectbox("Profile by", ["Hour of Day", "Day of Week", "Month", "Quarter", "Year"])
        df_profile = df_selected.copy()
        
        if profile_unit == "Hour of Day":
            df_profile['profile'] = df_profile.index.hour
        elif profile_unit == "Day of Week":
            df_profile['profile'] = df_profile.index.dayofweek
        elif profile_unit == "Month":
            df_profile['profile'] = df_profile.index.month
        elif profile_unit == "Quarter":
            df_profile['profile'] = df_profile.index.quarter
        else:
            df_profile['profile'] = df_profile.index.year

        # Always show aggregation method selection
        profile_agg_method = st.selectbox(
            "Aggregation Method",
            ["Mean", "Median", "Sum", "Min", "Max", "Std"],
            key="profile_agg_method"
        )
        
        plot_type = st.radio("Plot type", ["Boxplot", "Lineplot"], horizontal=True)
        
        if plot_type == "Boxplot":
            fig = px.box(df_profile, x='profile', y=selected_col, title=f"Boxplot by {profile_unit}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Calculate summary statistics using selected aggregation method
            agg_func = profile_agg_method.lower()
            profile_summary = df_profile.groupby('profile')[selected_col].agg([agg_func, 'std'])
            profile_summary.columns = ['value', 'std']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=profile_summary.index, 
                y=profile_summary['value'], 
                mode='lines+markers', 
                name=profile_agg_method
            ))
            fig.add_trace(go.Scatter(
                x=profile_summary.index, 
                y=profile_summary['value'] + profile_summary['std'], 
                mode='lines', 
                name='Upper', 
                line=dict(dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=profile_summary.index, 
                y=profile_summary['value'] - profile_summary['std'], 
                mode='lines', 
                name='Lower', 
                line=dict(dash='dot')
            ))
            fig.update_layout(
                title=f"Profile by {profile_unit} ({profile_agg_method})",
                xaxis_title=profile_unit,
                yaxis_title=selected_col
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Calendar Heatmap":
        st.subheader("📆 Calendar Heatmap (Daily)")
        daily_series = df_selected[selected_col].resample('D').agg(aggregation_method.lower())
        daily_series = daily_series.fillna(0)
        daily_df = daily_series.reset_index()
        daily_df.columns = ['date', selected_col]

        if daily_df[selected_col].isna().all() or daily_df[selected_col].eq(daily_df[selected_col].iloc[0]).all():
            st.warning("Data contains only NaN or constant values, which may prevent the color bar from displaying.")
            return

        fig = calplot(
            daily_df,
            x="date",
            y=selected_col,
            colorscale="Viridis",
            showscale=True
        )
        fig.update_layout(
            title=f"Calendar Heatmap: {selected_col} ({aggregation_method})",
            coloraxis_colorbar=dict(
                title=selected_col,
                titlefont=dict(size=14),
                thickness=10,
                len=0.8,
                x=0.95,
                y=0.5,
                yanchor="middle",
                tickformat=".2f",
                ticks="outside"
            ),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Heatmap Profiles":
        st.subheader("📅 Time Profiles Heatmap")
        df_time = df_selected.copy()

        df_time['year'] = df_time.index.year
        df_time['month'] = df_time.index.month
        df_time['hour'] = df_time.index.hour
        df_time['weekday'] = df_time.index.dayofweek
        df_time['week'] = df_time.index.isocalendar().week

        profile_type = st.selectbox(
            "Select Profile Type",
            ["Monthly", "Hourly", "Yearly", "Weekday", "Week of Year"],
            key="heatmap_profile_type"
        )

        # Always show aggregation method selection for heatmaps
        heatmap_agg_method = st.selectbox(
            "Aggregation Method", 
            ["Mean", "Median", "Sum", "Min", "Max", "Std"],
            key="heatmap_agg_method"
        )
        agg_func = heatmap_agg_method.lower()

        if profile_type == "Monthly":
            st.subheader("📅 Monthly Heatmap")
            heatmap_data = df_time.groupby(['year', 'month'])[selected_col].agg(agg_func).unstack()
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Month", y="Year", color=selected_col),
                x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title=f"Monthly Heatmap ({heatmap_agg_method})",
                xaxis_title="Month",
                yaxis_title="Year",
                coloraxis_colorbar=dict(
                    title=selected_col,
                    thickness=20,
                    len=0.8,
                    x=1.,
                    y=0.5,
                    yanchor="middle",
                    tickformat=".2f",
                    ticks="outside"
                )
            )

        elif profile_type == "Hourly":
            st.subheader("⏰ Hourly Heatmap")
            heatmap_data = df_time.groupby(['year', 'hour'])[selected_col].agg(agg_func).unstack()
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Hour", y="Year", color=selected_col),
                x=[f"{h:02d}:00" for h in range(24)],
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title=f"Hourly Heatmap ({heatmap_agg_method})",
                xaxis_title="Hour of Day",
                yaxis_title="Year",
                coloraxis_colorbar=dict(
                    title=selected_col,
                    thickness=20,
                    len=0.8,
                    x=1.,
                    y=0.5,
                    yanchor="middle",
                    tickformat=".2f",
                    ticks="outside"
                )
            )

        elif profile_type == "Yearly":
            st.subheader("📅 Yearly Heatmap")
            heatmap_data = df_time.groupby(['year'])[selected_col].agg(agg_func).to_frame().T
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Year", y="", color=selected_col),
                x=heatmap_data.columns,
                y=["Value"],
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title=f"Yearly Heatmap ({heatmap_agg_method})",
                xaxis_title="Year",
                yaxis_title="",
                coloraxis_colorbar=dict(
                    title=selected_col,
                    thickness=20,
                    len=0.8,
                    x=1.,
                    y=0.5,
                    yanchor="middle",
                    tickformat=".2f",
                    ticks="outside"
                )
            )

        elif profile_type == "Weekday":
            st.subheader("📅 Weekday Heatmap")
            heatmap_data = df_time.groupby(['year', 'weekday'])[selected_col].agg(agg_func).unstack()
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Day of Week", y="Year", color=selected_col),
                x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title=f"Weekday Heatmap ({heatmap_agg_method})",
                xaxis_title="Day of Week",
                yaxis_title="Year",
                coloraxis_colorbar=dict(
                    title=selected_col,
                    thickness=20,
                    len=0.8,
                    x=1.,
                    y=0.5,
                    yanchor="middle",
                    tickformat=".2f",
                    ticks="outside"
                )
            )

        elif profile_type == "Week of Year":
            st.subheader("📅 Week of Year Heatmap")
            heatmap_data = df_time.groupby(['year', 'week'])[selected_col].agg(agg_func).unstack()
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Week of Year", y="Year", color=selected_col),
                x=list(range(1, 54)),
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title=f"Week of Year Heatmap ({heatmap_agg_method})",
                xaxis_title="Week of Year",
                yaxis_title="Year",
                coloraxis_colorbar=dict(
                    title=selected_col,
                    thickness=20,
                    len=0.8,
                    x=1.,
                    y=0.5,
                    yanchor="middle",
                    tickformat=".2f",
                    ticks="outside"
                )
            )

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Forecasting":
        from prophet import Prophet

        st.subheader(f"📈 Forecasting for {selected_col}")
        
        df_prophet = df_selected.reset_index()
        first_column_name = df_prophet.columns[0]
        df_prophet = df_prophet.rename(columns={first_column_name: 'ds', selected_col: 'y'})

        if time_granularity != "None":
            df_prophet = df_prophet.set_index('ds').resample(resample_map[time_granularity]).agg({'y': aggregation_method.lower()}).reset_index()
        df_prophet = df_prophet[['ds', 'y']].copy()
        df_prophet['y'] = df_prophet['y'].fillna(method='ffill')

        if df_prophet.empty or df_prophet['ds'].isna().all() or df_prophet['y'].isna().all():
            st.error("No valid data available for forecasting. Please check the dataset.")
            return

        try:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
            model.fit(df_prophet)
        except ValueError as e:
            st.error(f"Error fitting Prophet model: {e}")
            return

        unit_map = {
            "Hourly": "hours",
            "Daily": "days",
            "Weekly": "weeks",
            "Monthly": "months",
            "Quarterly": "quarters",
            "Yearly": "years",
            "None": "days"
        }
        unit = unit_map.get(time_granularity, "days")

        if time_granularity == "None":
            forecast_period = st.slider(f"Select forecast period ({unit})", 1, 365, 30, format="%d")
        elif time_granularity == "Hourly":
            forecast_period = st.slider(f"Select forecast period ({unit})", 24, 8760, 720, format="%d")
        elif time_granularity == "Daily":
            forecast_period = st.slider(f"Select forecast period ({unit})", 7, 365, 30, format="%d")
        elif time_granularity == "Weekly":
            forecast_period = st.slider(f"Select forecast period ({unit})", 1, 52, 4, format="%d")
        elif time_granularity == "Monthly":
            forecast_period = st.slider(f"Select forecast period ({unit})", 1, 12, 3, format="%d")
        elif time_granularity == "Quarterly":
            forecast_period = st.slider(f"Select forecast period ({unit})", 1, 4, 1, format="%d")
        elif time_granularity == "Yearly":
            forecast_period = st.slider(f"Select forecast period ({unit})", 1, 5, 1, format="%d")

        future = model.make_future_dataframe(periods=forecast_period, freq=resample_map.get(time_granularity, 'D'))
        forecast = model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Historical Data', mode='lines'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', mode='lines'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', mode='lines', line=dict(color='rgba(0,0,0,0)')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', mode='lines', line=dict(color='rgba(0,0,0,0)')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)', name='Confidence Interval'))
        
        fig.update_layout(
            title=f"Forecast for {selected_col} ({aggregation_method if time_granularity != 'None' else 'Raw'})",
            xaxis_title="Date",
            yaxis_title=selected_col,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Changepoint Analysis":
        from prophet import Prophet
        import numpy as np
        import pandas as pd

        st.subheader("🔍 Changepoint Analysis")

        mode = st.radio("Choose changepoint mode", [
            "Default (Prophet auto-detection)",
            "Specify Number of Changepoints"
        ])

        df_prophet = df_selected.reset_index()
        first_column_name = df_prophet.columns[0]
        df_prophet = df_prophet.rename(columns={first_column_name: 'ds', selected_col: 'y'})

        if not pd.api.types.is_datetime64_any_dtype(df_prophet['ds']):
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], errors='coerce')
            if df_prophet['ds'].isna().all():
                st.error("No valid datetime values in 'ds' column. Please check the dataset.")
                return

        if time_granularity != "None":
            df_prophet = df_prophet.set_index('ds').resample(resample_map[time_granularity]).agg({'y': aggregation_method.lower()}).reset_index()

        df_prophet = df_prophet[['ds', 'y']].copy()
        df_prophet['y'] = df_prophet['y'].fillna(method='ffill')

        if df_prophet.empty or df_prophet['ds'].isna().all() or df_prophet['y'].isna().all():
            st.error("No valid data available for changepoint analysis. Please check the dataset.")
            return

        n_changepoints = 25
        show_threshold = False

        if mode == "Default (Prophet auto-detection)":
            st.info("✅ Using Prophet's default changepoint detection (25 automatically positioned).")
            show_threshold = True
        elif mode == "Specify Number of Changepoints":
            n_changepoints = st.slider("🔢 Number of changepoints to force", 1, 50, 10)
            show_threshold = True

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            n_changepoints=n_changepoints,
            seasonality_mode='additive'
        )

        try:
            model.fit(df_prophet)
            forecast = model.predict(df_prophet)

            deltas = np.array(model.params['delta']).mean(axis=0)
            abs_deltas = np.abs(deltas)

            if show_threshold:
                threshold = st.slider("⚠️ Slope change threshold (|Δ| > ...)", 0.0, 0.5, 0.01, 0.01)
            else:
                threshold = 0.0

            significant_cp = model.changepoints[abs_deltas > threshold] if threshold > 0.0 else model.changepoints
            plot_cp_mode = st.selectbox(
                "📌 Which changepoints to plot?",
                ["Only significant changepoints (|Δ| > threshold)", "All changepoints"]
            )
            changepoints_to_plot = significant_cp if plot_cp_mode.startswith("Only") else model.changepoints

            if len(changepoints_to_plot) == 0:
                st.warning("⚠️ No changepoints to plot with current settings.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name="Observed", mode='lines'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", mode='lines', line=dict(color='lightblue')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], name="Trend", mode='lines', line=dict(color='orange', dash='dash')))

            for cp in changepoints_to_plot:
                fig.add_vline(x=pd.to_datetime(cp), line=dict(color="red", dash="dot"))

            fig.update_layout(
                title="Changepoint Detection with Prophet (Trend, Forecast & Selected Δ)",
                xaxis_title="Date",
                yaxis_title=selected_col,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_range=[df_prophet['ds'].min(), df_prophet['ds'].max()]
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**🧠 {len(changepoints_to_plot)} changepoints plotted**")

            with st.expander(f"📋 List of Significant Changepoints (|Δslope| > {threshold})"):
                if len(significant_cp) > 0:
                    cp_df = pd.DataFrame({
                        "Changepoint": pd.to_datetime(significant_cp),
                        "|Δslope|": abs_deltas[abs_deltas > threshold]
                    }).reset_index(drop=True)
                    st.dataframe(cp_df)
                else:
                    st.warning("No changepoints exceed the threshold.")

            with st.expander("📊 Distribution of |Δslope| values across changepoints"):
                fig_d = go.Figure()
                fig_d.add_trace(go.Histogram(x=abs_deltas, nbinsx=20))
                fig_d.update_layout(
                    xaxis_title="|Δslope|", yaxis_title="Frequency",
                    title="Distribution of Slope Changes",
                    bargap=0.2
                )
                st.plotly_chart(fig_d, use_container_width=True)

            with st.expander("🧪 All changepoints and slope changes (Δ)"):
                all_cp_df = pd.DataFrame({
                    "Changepoint": pd.to_datetime(model.changepoints),
                    "Δslope": deltas,
                    "|Δslope|": abs_deltas
                }).reset_index(drop=True)
                st.dataframe(all_cp_df.sort_values(by="|Δslope|", ascending=False))

        except Exception as e:
            st.error(f"Changepoint detection failed: {e}")

    elif analysis_section == "Trend Analysis":
        st.subheader("📈 Trend Analysis")
    
        df_trend = df_summary.copy()
        x = df_trend.index
        y = df_trend['value']
    
        from sklearn.linear_model import LinearRegression
        import numpy as np
        from sklearn.preprocessing import PolynomialFeatures
        from statsmodels.tsa.seasonal import STL
    
        x_numeric = np.arange(len(x)).reshape(-1, 1)
        y_numeric = y.values.reshape(-1, 1)
    
        model_linear = LinearRegression()
        model_linear.fit(x_numeric, y_numeric)
        trend_linear = model_linear.predict(x_numeric).flatten()
        slope = model_linear.coef_[0][0]
        intercept = model_linear.intercept_[0]
    
        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x_numeric)
        model_poly = LinearRegression()
        model_poly.fit(x_poly, y_numeric)
        trend_poly = model_poly.predict(x_poly).flatten()
    
        stl = STL(y, period=12)
        res = stl.fit()
        seasonal = res.seasonal
        deseasonalized = y - seasonal
    
        model_linear_deseason = LinearRegression()
        model_linear_deseason.fit(x_numeric, deseasonalized.values.reshape(-1, 1))
        trend_linear_deseason = model_linear_deseason.predict(x_numeric).flatten()
        slope_deseason = model_linear_deseason.coef_[0][0]
        intercept_deseason = model_linear_deseason.intercept_[0]
    
        poly_deseason = PolynomialFeatures(degree=2)
        x_poly_deseason = poly_deseason.fit_transform(x_numeric)
        model_poly_deseason = LinearRegression()
        model_poly_deseason.fit(x_poly_deseason, deseasonalized.values.reshape(-1, 1))
        trend_poly_deseason = model_poly_deseason.predict(x_poly_deseason).flatten()
    
        show_deseasonalized = st.checkbox("Plot deseasonalized line/trend", value=False, key="trend_toggle")
    
        if show_deseasonalized:
            summary_data = {
                'Trend Type': ['Linear (Deseasonalized)', 'Polynomial (Deseasonalized)'],
                'Min': [trend_linear_deseason.min(), trend_poly_deseason.min()],
                'Max': [trend_linear_deseason.max(), trend_poly_deseason.max()]
            }
        else:
            summary_data = {
                'Trend Type': ['Linear', 'Polynomial'],
                'Min': [trend_linear.min(), trend_poly.min()],
                'Max': [trend_linear.max(), trend_poly.max()]
            }
    
        import plotly.graph_objects as go
        fig = go.Figure()
    
        if show_deseasonalized:
            fig.add_trace(go.Scatter(x=x, y=deseasonalized, mode='lines', name='Deseasonalized'))
            fig.add_trace(go.Scatter(x=x, y=trend_linear_deseason, mode='lines', name='Linear Trend (Deseasonalized)', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=x, y=trend_poly_deseason, mode='lines', name='Polynomial Trend (Deseasonalized)', line=dict(dash='dot')))
            y_max_for_annotation = deseasonalized.max()
            regression_text = f'Linear (Deseasonalized): y = {slope_deseason:.8f}x + {intercept_deseason:.8f}'
        else:
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'Observed ({aggregation_method})'))
            fig.add_trace(go.Scatter(x=x, y=trend_linear, mode='lines', name='Linear Trend', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=x, y=trend_poly, mode='lines', name='Polynomial Trend', line=dict(dash='dot')))
            y_max_for_annotation = y.max()
            regression_text = f'Linear: y = {slope:.8f}x + {intercept:.8f}'
    
        if show_deseasonalized:
            y_linear_min = trend_linear_deseason.min()
            y_linear_max = trend_linear_deseason.max()
            y_poly_min = trend_poly_deseason.min()
            y_poly_max = trend_poly_deseason.max()
            y_poly_first = trend_poly_deseason[0]
            y_poly_last = trend_poly_deseason[-1]
        else:
            y_linear_min = trend_linear.min()
            y_linear_max = trend_linear.max()
            y_poly_min = trend_poly.min()
            y_poly_max = trend_poly.max()
            y_poly_first = trend_poly[0]
            y_poly_last = trend_poly[-1]
    
        fig.add_annotation(x=x[trend_linear.argmin()], y=y_linear_min, text=f'Min: {y_linear_min:.2f}', showarrow=True, arrowhead=1, ax=20, ay=-30)
        fig.add_annotation(x=x[trend_linear.argmax()], y=y_linear_max, text=f'Max: {y_linear_max:.2f}', showarrow=True, arrowhead=1, ax=20, ay=30)
        fig.add_annotation(x=x[trend_poly.argmin()], y=y_poly_min, text=f'Poly Min: {y_poly_min:.2f}', showarrow=True, arrowhead=1, ax=-20, ay=-30)
        fig.add_annotation(x=x[trend_poly.argmax()], y=y_poly_max, text=f'Poly Max: {y_poly_max:.2f}', showarrow=True, arrowhead=1, ax=-20, ay=30)
        fig.add_annotation(x=x[0], y=y_poly_first, text=f'Poly First: {y_poly_first:.2f}', showarrow=True, arrowhead=1, ax=20, ay=0)
        fig.add_annotation(x=x[-1], y=y_poly_last, text=f'Poly Last: {y_poly_last:.2f}', showarrow=True, arrowhead=1, ax=-20, ay=0)
    
        fig.add_annotation(
            x=x[0], y=y_max_for_annotation,
            xref="x", yref="y",
            text=regression_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(0, 0, 0, 0.7)",
            font=dict(color="white", size=12),
            bordercolor="black",
            borderwidth=1
        )
    
        fig.update_layout(
            title=f"Trend Analysis for {selected_col} ({aggregation_method} with {time_granularity} granularity)",
            xaxis_title="Date",
            yaxis_title=selected_col,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_range=[x.min(), x.max()]
        )
    
        st.plotly_chart(fig, use_container_width=True)
        st.table(pd.DataFrame(summary_data))