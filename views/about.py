import streamlit as st

def about():

    with st.sidebar:
        st.markdown("""
        ### ℹ️ About this App

        Explore your datasets **interactively**:

        - 📊 Create rich visualizations (histograms, boxplots, density plots, regressions).
        - 🧹 Process data (filter, melt, create date features, subsample).
        - 🧩 Analyze missing values (matrix, bars, heatmap, dendrogram).
        - 🎛️ Customize plots with hue, facetting, category orders, and more.
        - 💾 Download processed data easily.

        Built with **Streamlit** and **Plotly** for fast and flexible **data exploration**!
        """)

    st.subheader("📖 About This App")

    st.markdown("""Welcome to the **Interactive Data Visualization App**!  
    A powerful and flexible tool to explore, preprocess, and visualize your datasets easily.""")



    with st.expander("ℹ️ User manual", expanded=False):
        st.markdown("""
        ### 📘 Manual: Interactive Data Visualization App

        #### 1. 📂 Uploading and Preparing Data

        ➔ **Step 1. Upload Your CSV**  
        📎 Go to the Home section.  
        ➡️ Click: **Upload CSV** and select your dataset.  
        ![Upload CSV](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/browse_myvp.png)

        ➔ **Step 2. Review and Select Columns**  
        ✅ After upload:  
        - Review column types (Numeric/Categorical).
        - Select columns to keep.  

        ➔ **Step 3. Assign Variable Types**  
        ⚙️ Assign manually:  
        - Numeric variables  
        - Categorical variables  

        ➔ **Step 4. Optional Preprocessing**  
        - Subsample (% of data)  
        - Extract Date Features (Year, Month, Day, Hour)  
        - Melt Data (reshape wide to long)  
        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/prepro_myvp.png)

        ➔ **Step 5. Preview and Download**  
        📄 Preview the processed dataset.  
        💾 Download your cleaned CSV.  
        ![Insert Screenshot: Upload CSV](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/sample_download_myvp.png)

        ➔ **Recommendation:** not all operations can be applied at once, try this:  
        ⚠️ Load ➔ apply operation ➔ check preview ➔ download ➔ repeat.

        ---

        #### 2. 📊 Visualization Options

        | Plot Type | Purpose |
        |:---|:---|
        | 📊 Bars | Bar plots or time-indexed bars. |
        | 📉 Histogram | Distribution of numerical variables. |
        | 🌈 Density 1 | Univariate density plots (layer/stacked). |
        | 🌐 Density 2 | 2D Density (Histograms or Contours). |
        | 🎁 Boxes | Boxplots, Lineplots, Violin plots. |
        | 🏔️ Ridgeplots | Density ridges by categories or numeric. |
        | 📈 Regression | Scatter + Regression lines + Confidence bands. |
        | 🧩 Pairplots | Scatter matrix with KDE diagonals. |
        | ❔ Missingno | Missing data patterns visualization. |

        ---

        #### 3. ⚙️ Controls and Customization

        - **X Variable**: Variable for X-axis.
        - **Y Variable**: Variable for Y-axis (if needed).
        - **Hue (Color)**: Color split (optional).
        - **Facet Column / Row**: Create multi-panel plots.

        🔧 **Plot-specific Settings**:
        - Histograms: Step / Bar / Layer / Stack
        - Densities: Cumulative / Normalize
        - Boxplots/Violins: Swarm points
        - Lineplots: Confidence bands ±1 std
        - Regression: Polynomial orders (1, 2, 3) and CI (68%, 90%, 95%, 99%)

        🔠 **Custom Order**:
        - Sort categories (X, Hue, Facet) manually.

        ---

        #### 4. 🧩 Special Behaviors

        🛡️ **Auto Handling**:
        - Too many categories (>100): Falls back to numeric ridges.

        🛠️ **Missing Data Tools**:
        - Matrix view of missingness
        - Missingness correlation heatmap
        - Dendrogram based on missing patterns

        ---

        #### 5. 🚀 Tips for Best Use

        - 📈 Start with Histograms and Densities to understand distributions.
        - 🎯 Keep Facets manageable (no more than 5-10 categories).
        - 🧹 Use Missing Data Heatmaps to find important patterns.
        - 🔠 Use Custom Ordering for ordinal variables (months, years, etc).
        - 💾 Download your cleaned dataset after transformations.

        ---

        #### ✨ End of Manual

        If needed, re-upload your dataset anytime to start fresh.  
        _Enjoy exploring your data visually! 🎨_

        ---

        ### Examples: load the suggested datasets, check the plots below and the configuration in the left panel:

        - Bars (use [tips dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Tips.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/bars_myvp.png)

        - Boxes (use [tips dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Tips.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/boxes_myvp.png)

        - Ridges (use [tips dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Tips.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/ridges_myvp.png)

        - Histograms (use [tips dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Tips.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/hist_myvp.png)

        - Density 1 (use [tips dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Tips.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/kde_myvp.png)

        - Density 2 (use [tips dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Tips.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/kde2_myvp.png)

        - Scatter (use [tips dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Tips.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/scatter_myvp.png)

        - Missing values - missingno ([penguins dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Penguins.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/missingno_myvp.png)

        - Cluster map (similar to correlation map, but variables are grouped by correlation, use [penguins dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Penguins.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/clustermap_myvp.png)

        - Pairplot (use [penguins dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Penguins.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/pair_myvp.png)

        - Regression ([penguins dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Penguins.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/reg_myvp.png)

        - Ridges (all columns numerical, remove columns in preprocessing from the [time series dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/ts_data_precipitation.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/ridges_num_myvp.png)

        - Heatmap (all columns numerical, remove columns in preprocessing from the [time series dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/ts_data_precipitation.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/heatnum_myvp.png)

        - Time series as bars (keep column FECHA from the [time series dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/ts_data_precipitation.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/barsts_myvp.png)

        - Time series as lines (keep column FECHA from the [time series dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/ts_data_precipitation.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/linests_myvp.png)

        - Catplot (use [tips dataset](https://github.com/marsgr6/r-scripts/blob/master/data/viz_data/Tips.csv)):

        ![](https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/imgs/catplot_myvp.png)

        """)

    st.markdown("""

### 🔥 Key Features

- **Data Upload and Preprocessing**
  - Upload CSV files.
  - Assign variable types (numeric, categorical).
  - Subsample data randomly.
  - Extract date features (Year, Month, Day, Hour).
  - Reshape datasets with melting.
  - Download the processed dataset.

- **Data Requirements**:
  - Supports both categorical and numerical data.
  - Time series mode requires an object column (e.g., dates) for the x-axis and a numerical column for the y-axis.
  - Raw Heatmap mode requires either all numerical columns or a categorical and numerical variable pair.
  - Warnings are displayed if data types don’t meet requirements for specific modes.

- **Visualizations**
  - **Univariate**: Histograms, KDE plots, Boxplots, Violin plots.
  - **Bivariate**: Scatter plots, Regression plots with confidence intervals.
  - **Multivariate**: Pairplots, Ridgeplots, Heatmaps.
  - **Missing Data**: Visualize missingness, correlations, dendrograms.

- **Time Series Support (very limited)**:
  - Enable the "Time is here" option in line plots and bar plots to plot raw time series data (e.g., dates on the x-axis and numerical values on the y-axis) without aggregation or sorting.

- **Advanced Controls**
  - Custom order for categorical variables (X, Hue, Facets).
  - Faceting by row and column.
  - Optional swarm overlay for boxplots and violins.
  - Display statistical bands (±1 std) in line plots.
  - Cumulative and normalized density options.

- **Dynamic Behavior**
  - Automatically adjusts when only numeric or only categorical data is present.
  - Fallback strategies when too many categories (>100) are detected.
  - Responsive and interactive plotting with Plotly.

### 🚀 Designed For

- **Exploratory Data Analysis (EDA)**
- **Data quality assessment** (missing data structures)
- **Educational projects** in Data Science and Statistics
- **Quick visualization prototyping**


### 🌟 Why This App?

- *Every dataset tells a story — make it a good one!*

### ⚠️ Important Note

- 🛠️ **This app is still under active development and may contain bugs.**

- Some functionalities (especially **complex faceting** and **dynamic interactions**) might produce unexpected behavior under edge cases (e.g., too many categories, missing values, duplicated variables in facets).

- Please **reload** the page if the app becomes unstable, and **re-upload your dataset** for a fresh start. 📂✨

- Your feedback is welcome to continue improving it! 🚀

    """)