import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from datetime import date

# PAGE CONFIGURATION
st.set_page_config(page_title="Loanboox Analytics Dashboard", layout="wide")

relevant_cols = ['5Y_SARON_swap_fixed_rate', '3m_SARON_OIS_rate', 'Curve', 'SNB_Leitzins', 'Inflation', 'M2', 'Req_Volume', 'Selected_Volume']

# Loanboox Branding: this comes from ChatGPT
st.markdown("""
    <style>
        :root { --primary-color: #004B66; --secondary-color: #00A6D6; }
        h1, h2, h3, .css-10trblm { color: #004B66 !important; }
        div[data-testid="stMetricLabel"] { color: #555555; }
        div[data-testid="stMetricValue"] { color: #004B66; }
        button[data-baseweb="tab"] { color: #555555; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #004B66; border-color: #004B66; }
    </style>
""", unsafe_allow_html=True)

# COLOURS
COLOR_PRIMARY = "#004B66" 
COLOR_SECONDARY = "#00A6D6"
BLACK = "#000000"

# Data load and preprocessing function (date)
# I use st.cache_data to cache the loaded data for more speed as you suggested
@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the data from the uploaded Excel file."""
    # We assume that it is always an Excel file. 
    df = pd.read_excel(uploaded_file)

    # Check for required columns
    required_columns = ['datum'] + relevant_cols
    
    # If required columns are missing, show error and stop message and stop execution
    if not set(required_columns).issubset(df.columns):
        st.error("Wrong dataset uploaded. Please upload the cleaned Loanboox Excel file from the Jupyter Notebook.")
        st.stop()

    # Direct date conversion (“date” exists in the dataset, as it is cleaned beforehand)
    df['datum'] = pd.to_datetime(df['datum'])
    df['YearMonth'] = df['datum'].dt.to_period('M').astype(str)
    
    return df

# Main application

def main():
    st.title("Loanboox Data Explorer")
    st.markdown("Visualize financial data and explore it interactively!")
    
    # Logo
    st.sidebar.image("logo.png", use_container_width=True)

    # File uploader
    uploaded_file = st.file_uploader("Select the cleaned Loanboox Excel file", type=['xlsx', 'xls'])

    # stop if no file is uploaded
    if not uploaded_file:
        st.stop()

    # load data
    df_raw = load_data(uploaded_file)
    
    # sidebar title and settings
    st.sidebar.header("Settings")
    
    # data cleaning option
    use_clean_data = st.sidebar.checkbox("Clean data (1 row per request)", value=True)
    
    # filter date range 
    st.sidebar.subheader("Date Filter")
    min_date = df_raw['datum'].min().date()
    max_date = df_raw['datum'].max().date()
    
    # As the company grew until 2018, we set fixed limits
    default_start = max(date(2018, 1, 1), min_date)
    default_end = min(date(2025, 9, 30), max_date)
    
    start_date = st.sidebar.date_input("Start date", default_start, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", default_end, min_value=min_date, max_value=max_date)

    # Apply date filter
    mask = (df_raw['datum'].dt.date >= start_date) & (df_raw['datum'].dt.date <= end_date)
    df_filtered = df_raw.loc[mask]

    # Data filtering options
    st.sidebar.header("Data Filtering Options")
    
    # save the state of the checkboox
    if 'filter_zeros' not in st.session_state:
        st.session_state.filter_zeros = False

    manual_filter_zeros = st.sidebar.checkbox("Only successful closings (selected_volume > 0)", key='filter_zeros')
    
    do_agg = st.sidebar.checkbox("Aggregate monthly", value=False)
    do_log = st.sidebar.checkbox("Logarithmize Y-variables", value=False)

    # data cleaning (exact same logic as in the Jupyter Notebook)
    if use_clean_data:
        df_sorted = df_filtered.sort_values(by=["OfferRequestId", "Selected_Volume"], ascending=[True, False])
        df_working = df_sorted.drop_duplicates(subset=["OfferRequestId"]).copy()
    else:
        df_working = df_filtered

    # filter successful closings (>0)
    if manual_filter_zeros:
        df_working_filtered = df_working[df_working['Selected_Volume'] > 0]
    else:
        df_working_filtered = df_working

    st.divider()

    # Cool looking dashboard metrics with big numbers
    st.subheader("Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Number of records", len(df_working_filtered))
    col2.metric("Total request volume", f"CHF {df_working_filtered['Req_Volume'].sum()/1e6:,.1f} M")
    col3.metric("Successful volume", f"CHF {df_working_filtered['Selected_Volume'].sum()/1e6:,.1f} M")
 

    st.divider()

    # tabs on the website
    tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "General Analysis", "Hypothesis Tester", "Data"])

    # TAB 1: Time Series
    with tab1:
        st.subheader("Volume over time")
        df_plot_time = df_working_filtered.copy()
        
        if do_agg:
            df_daily = df_plot_time.groupby('YearMonth')[['Req_Volume', 'Selected_Volume']].sum().reset_index()
            x_axis = 'YearMonth'
        else:
            df_daily = df_plot_time.groupby('datum')[['Req_Volume', 'Selected_Volume']].sum().reset_index()
            x_axis = 'datum'
        
        if do_log:
            df_daily['Req_Volume'] = np.log1p(df_daily['Req_Volume'])
            df_daily['Selected_Volume'] = np.log1p(df_daily['Selected_Volume'])
            title_suffix = " (Log Scale)"
        else:
            title_suffix = ""

        fig_line = px.line(df_daily, x=x_axis, y=['Req_Volume', 'Selected_Volume'],
                           title=f"Volume development{title_suffix}",
                           color_discrete_map={"Req_Volume": COLOR_PRIMARY, "Selected_Volume": COLOR_SECONDARY})
        st.plotly_chart(fig_line, use_container_width=True)

    # TAB 2: Correlation & Macro
    with tab2:
        st.subheader("Correlation Matrix")
        
            
        # checkbox selection for correlation matrix
        st.markdown("#### Select variables for correlation matrix:")
        
        cols = st.columns(4)
        selected_vars_matrix = []
        
        # selection of important variables from relecvant_cols
        default_selection = ['5Y_SARON_swap_fixed_rate', '3m_SARON_OIS_rate', 'Curve', 'SNB_Leitzins', 'Inflation', 'M2']

        for i, var in enumerate(relevant_cols):
            if cols[i % 4].checkbox(var, value=(var in default_selection), key=f"corr_{var}"):
                selected_vars_matrix.append(var)
        
        if len(selected_vars_matrix) > 1:
            df_corr = df_working_filtered[selected_vars_matrix].copy()
            if do_log:
                for col in ['Req_Volume', 'Selected_Volume']:
                    if col in df_corr.columns: df_corr[col] = np.log1p(df_corr[col])
            
            fig_corr = px.imshow(df_corr.corr(), text_auto=".2f", aspect="auto", color_continuous_scale=[BLACK, COLOR_SECONDARY], origin='lower')
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Please select at least two variables.")

        st.divider()
        
        # Checkbox selection for macro trends
        st.subheader("Macro data trends")
        st.markdown("#### Select macro variables to plot:")
        
        macro_cols_all = ['5Y_SARON_swap_fixed_rate', '3m_SARON_OIS_rate', 'Curve', 'SNB_Leitzins', 'Inflation', 'M2']
        default_macro = ['5Y_SARON_swap_fixed_rate', 'Inflation']
        
        cols_macro = st.columns(4)
        selected_exog_plot = []
        
        for i, var in enumerate(macro_cols_all):
            if cols_macro[i % 4].checkbox(var, value=(var in default_macro), key=f"macro_{var}"):
                selected_exog_plot.append(var)
        
        if selected_exog_plot:
            df_macro_time = df_working_filtered.groupby('datum')[selected_exog_plot].mean().reset_index()
            for var in selected_exog_plot:
                fig_macro = px.line(df_macro_time, x='datum', y=var, title=f"Trend: {var}")
                fig_macro.update_traces(line_color=COLOR_PRIMARY)
                st.plotly_chart(fig_macro, use_container_width=True)
        else:
            st.info("Please select at least one macro variable.")

    # TAB 3: Hypothesis Tester
    with tab3:
        st.subheader("Verify Hypotheses")
        
        pot_y = ['Req_Volume', 'Selected_Volume'] # , 'Anzahl_Unt_Lender' can be added but is to complicated for now (requires more complicated regression)
        pot_x = ['5Y_SARON_swap_fixed_rate', '3m_SARON_OIS_rate', 'Curve', 'SNB_Leitzins', 'Inflation', 'M2']

        c1, c2 = st.columns(2)
        
        # If the selection changes, update the filter_zeros state
        def on_y_change():
            if st.session_state.y_var_select == 'Selected_Volume':
                st.session_state.filter_zeros = True
            else:
                st.session_state.filter_zeros = False

        y_var = c1.selectbox("Y-Axis (Target)", pot_y, key='y_var_select', on_change=on_y_change)
        x_var = c2.selectbox("X-Axis (Factor)", pot_x)

        # Prepare data for plotting     
        df_plot = df_working_filtered.copy()
        
        # Aggregation
        if do_agg:
            agg_type = 'sum' if y_var in ['Req_Volume', 'Selected_Volume'] else 'mean'
            df_plot = df_plot.groupby('YearMonth').agg({x_var: 'mean', y_var: agg_type}).reset_index()
        
        # Logarithmization
        if do_log:
            df_plot[y_var] = np.log1p(df_plot[y_var])
            y_label = f"Log({y_var})"
        else:
            y_label = y_var

        # Plot
        if not df_plot.empty:
            fig_hypo = px.scatter(df_plot, x=x_var, y=y_var, trendline="ols",
                                  title=f"Influence of {x_var} on {y_label}",
                                  labels={y_var: y_label}, color_discrete_sequence=[COLOR_PRIMARY])
            st.plotly_chart(fig_hypo, use_container_width=True)
            
            # regression Analysis
            if len(df_plot) > 1:
                st.divider()
                st.markdown("### OLS Regression Analysis")
                
                # 1. Clean Data: drop NaN values
                df_reg = df_plot[[x_var, y_var]].dropna()
                
                # 2. Prepare megression
                Y = df_reg[y_var]
                # adding a constant through add_constant
                X = sm.add_constant(df_reg[x_var])
                
                # 3. Fit model
                model = sm.OLS(Y, X).fit()
                
                # 4. Extract Results 
                r2 = model.rsquared          # How well does X explain Y?
                coef = model.params[x_var]   # Slope 
                p_value = model.pvalues[x_var] # Significance
                
                # 5. Display Metrics
                c1, c2, c3 = st.columns(3)
                
                # R-Squared
                c1.metric(
                    "R² (Explanatory Power)", 
                    f"{r2:.4f}", 
                    help="Indicates the percentage of the variance in Y that is explained by X. 1.0 would be a perfect fit."
                )
                
                # Slope
                c2.metric(
                    "Coefficient", 
                    f"{coef:.4f}", 
                    help="Indicates how much Y changes when X increases by 1 unit."
                )
                
                # Significance (P-Value)
                p_fmt = f"{p_value:.4e}" if p_value < 0.001 else f"{p_value:.4f}"
                c3.metric(
                    "P-Value", 
                    p_fmt, 
                    help="The probability that the result is just random chance."
                )
                
                # Conclusion box
                if p_value < 0.05:
                    st.info(f"**Significant Result** (p < 0.05). The variable *{x_var}* has a statistically verifiable influence on *{y_var}*.")
                else:
                    st.warning(f"**No Significant Relationship** (p > 0.05). The influence of *{x_var}* on *{y_var}* cannot be statistically proven.")
                
                # Info text for less experienced users
                with st.expander("Methodology & Limitations"):
                    st.markdown("""
                    **What this analysis does:**
                    This tool performs a **Simple Linear Regression (OLS)**. It attempts to fit a straight line through the data points to determine if changes in the chosen *Factor (X)* reliably predict changes in the *Target (Y)*.

                    **Limitations:**
                    * **Correlation doesn't mean Causation:** A significant result proves a relationship, but it does not prove that X causes Y.
                    * **Linearity assumption:** This model assumes a straight-line relationship.
                    * **Single factor view:** This analysis looks at only one factor at a time. In reality, the target variable might be influenced by multiple factors simultaneously (e.g., Seasonality + Interest Rates). This has been conducted in the Jupyter Notebook.
                    """)
                

    # TAB 4: Raw Data to filter and play around
    with tab4:
        st.dataframe(df_working_filtered)

if __name__ == "__main__":
    main()