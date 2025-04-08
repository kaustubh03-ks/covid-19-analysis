import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import statsmodels.api as sm

# Page configuration
st.set_page_config(
    page_title="COVID-19 Data Analysis Dashboard",
    page_icon="🦠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the COVID-19 dataset"""
    df = pd.read_csv('covid_19_clean_complete.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def calculate_growth_rate(df, country=None):
    """Calculate daily growth rate of confirmed cases"""
    if country:
        data = df[df['Country/Region'] == country].groupby('Date')['Confirmed'].sum().reset_index()
    else:
        data = df.groupby('Date')['Confirmed'].sum().reset_index()
    
    data['Growth_Rate'] = data['Confirmed'].pct_change() * 100
    return data

def calculate_mortality_rate(df, country=None):
    """Calculate mortality rate (deaths per 100,000 population)"""
    # Using a simplified approach since we don't have population data
    # In a real analysis, you would use actual population data
    if country:
        data = df[df['Country/Region'] == country].groupby('Date').agg({
            'Confirmed': 'sum',
            'Deaths': 'sum'
        }).reset_index()
    else:
        data = df.groupby('Date').agg({
            'Confirmed': 'sum',
            'Deaths': 'sum'
        }).reset_index()
    
    # Calculate case fatality rate (deaths/confirmed)
    data['Case_Fatality_Rate'] = (data['Deaths'] / data['Confirmed']) * 100
    
    return data

def calculate_recovery_rate(df, country=None):
    """Calculate recovery rate"""
    if country:
        data = df[df['Country/Region'] == country].groupby('Date').agg({
            'Confirmed': 'sum',
            'Recovered': 'sum'
        }).reset_index()
    else:
        data = df.groupby('Date').agg({
            'Confirmed': 'sum',
            'Recovered': 'sum'
        }).reset_index()
    
    # Calculate recovery rate (recovered/confirmed)
    data['Recovery_Rate'] = (data['Recovered'] / data['Confirmed']) * 100
    
    return data

def calculate_active_case_ratio(df, country=None):
    """Calculate active case ratio"""
    if country:
        data = df[df['Country/Region'] == country].groupby('Date').agg({
            'Confirmed': 'sum',
            'Active': 'sum'
        }).reset_index()
    else:
        data = df.groupby('Date').agg({
            'Confirmed': 'sum',
            'Active': 'sum'
        }).reset_index()
    
    # Calculate active case ratio (active/confirmed)
    data['Active_Case_Ratio'] = (data['Active'] / data['Confirmed']) * 100
    
    return data

def main():
    st.title("🦠 COVID-19 Global Data Analysis Dashboard")
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Global Overview", "Country Analysis", "Epidemiological Analysis"])
    
    if page == "Global Overview":
        show_global_overview(df)
    elif page == "Country Analysis":
        show_country_analysis(df)
    else:
        show_epidemiological_analysis(df)

def show_global_overview(df):
    st.header("Global Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Confirmed Cases", f"{df['Confirmed'].max():,.0f}")
    with col2:
        st.metric("Total Deaths", f"{df['Deaths'].max():,.0f}")
    with col3:
        st.metric("Total Recovered", f"{df['Recovered'].max():,.0f}")
    with col4:
        st.metric("Active Cases", f"{df['Active'].max():,.0f}")
    
    # Global trends
    st.subheader("Global Trends")
    global_data = df.groupby('Date').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Recovered': 'sum',
        'Active': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=global_data['Date'], y=global_data['Confirmed'], name='Confirmed'))
    fig.add_trace(go.Scatter(x=global_data['Date'], y=global_data['Deaths'], name='Deaths'))
    fig.add_trace(go.Scatter(x=global_data['Date'], y=global_data['Recovered'], name='Recovered'))
    fig.update_layout(title="Global COVID-19 Cases Over Time", xaxis_title="Date", yaxis_title="Number of Cases")
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional analysis
    st.subheader("Regional Impact")
    regional_data = df.groupby('WHO Region').agg({
        'Confirmed': 'max',
        'Deaths': 'max',
        'Recovered': 'max',
        'Active': 'max'
    }).reset_index()
    
    fig = px.bar(regional_data, x='WHO Region', y=['Confirmed', 'Deaths', 'Recovered'],
                 title="COVID-19 Impact by WHO Region",
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Global rates
    st.subheader("Global Rates")
    mortality_data = calculate_mortality_rate(df)
    recovery_data = calculate_recovery_rate(df)
    active_data = calculate_active_case_ratio(df)
    
    # Latest rates
    latest_mortality = mortality_data['Case_Fatality_Rate'].iloc[-1]
    latest_recovery = recovery_data['Recovery_Rate'].iloc[-1]
    latest_active = active_data['Active_Case_Ratio'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Global Case Fatality Rate", f"{latest_mortality:.2f}%")
    with col2:
        st.metric("Global Recovery Rate", f"{latest_recovery:.2f}%")
    with col3:
        st.metric("Global Active Case Ratio", f"{latest_active:.2f}%")

def show_country_analysis(df):
    st.header("Country Analysis")
    
    # Country selector
    countries = sorted(df['Country/Region'].unique())
    selected_country = st.selectbox("Select Country", countries)
    
    # Country-specific metrics
    country_data = df[df['Country/Region'] == selected_country]
    latest_data = country_data.loc[country_data['Date'] == country_data['Date'].max()]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Confirmed Cases", f"{latest_data['Confirmed'].sum():,.0f}")
    with col2:
        st.metric("Total Deaths", f"{latest_data['Deaths'].sum():,.0f}")
    with col3:
        st.metric("Total Recovered", f"{latest_data['Recovered'].sum():,.0f}")
    with col4:
        st.metric("Active Cases", f"{latest_data['Active'].sum():,.0f}")
    
    # Country trends
    st.subheader(f"Trends for {selected_country}")
    country_trends = country_data.groupby('Date').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Recovered': 'sum',
        'Active': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=country_trends['Date'], y=country_trends['Confirmed'], name='Confirmed'))
    fig.add_trace(go.Scatter(x=country_trends['Date'], y=country_trends['Deaths'], name='Deaths'))
    fig.add_trace(go.Scatter(x=country_trends['Date'], y=country_trends['Recovered'], name='Recovered'))
    fig.update_layout(title=f"COVID-19 Cases in {selected_country}", xaxis_title="Date", yaxis_title="Number of Cases")
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth rate analysis
    st.subheader("Daily Growth Rate")
    growth_data = calculate_growth_rate(df, selected_country)
    fig = px.line(growth_data, x='Date', y='Growth_Rate',
                  title=f"Daily Growth Rate of Confirmed Cases in {selected_country}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Country-specific rates
    st.subheader(f"Rates for {selected_country}")
    mortality_data = calculate_mortality_rate(df, selected_country)
    recovery_data = calculate_recovery_rate(df, selected_country)
    active_data = calculate_active_case_ratio(df, selected_country)
    
    # Latest rates
    latest_mortality = mortality_data['Case_Fatality_Rate'].iloc[-1]
    latest_recovery = recovery_data['Recovery_Rate'].iloc[-1]
    latest_active = active_data['Active_Case_Ratio'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Case Fatality Rate", f"{latest_mortality:.2f}%")
    with col2:
        st.metric("Recovery Rate", f"{latest_recovery:.2f}%")
    with col3:
        st.metric("Active Case Ratio", f"{latest_active:.2f}%")

def show_epidemiological_analysis(df):
    st.header("Epidemiological Analysis")
    
    # Country selector
    countries = sorted(df['Country/Region'].unique())
    selected_country = st.selectbox("Select Country for Analysis", countries)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Case Fatality Analysis", "Recovery Analysis", "Active Case Analysis", "Comparative Analysis"])
    
    with tab1:
        st.subheader("Case Fatality Rate Analysis")
        mortality_data = calculate_mortality_rate(df, selected_country)
        
        # Plot case fatality rate over time
        fig = px.line(mortality_data, x='Date', y='Case_Fatality_Rate',
                      title=f"Case Fatality Rate Over Time in {selected_country}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest case fatality rate
        latest_cfr = mortality_data['Case_Fatality_Rate'].iloc[-1]
        st.metric("Latest Case Fatality Rate", f"{latest_cfr:.2f}%")
        
        # Explanation
        st.markdown("""
        <div class="highlight">
        <h4>What is Case Fatality Rate?</h4>
        <p>The Case Fatality Rate (CFR) is the proportion of deaths from a certain disease compared to the total number of people diagnosed with the disease for a particular period. 
        It is often expressed as a percentage and is calculated as:</p>
        <p>CFR = (Number of deaths / Number of confirmed cases) × 100</p>
        <p>A higher CFR indicates a more severe disease outcome.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Recovery Rate Analysis")
        recovery_data = calculate_recovery_rate(df, selected_country)
        
        # Plot recovery rate over time
        fig = px.line(recovery_data, x='Date', y='Recovery_Rate',
                      title=f"Recovery Rate Over Time in {selected_country}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest recovery rate
        latest_rr = recovery_data['Recovery_Rate'].iloc[-1]
        st.metric("Latest Recovery Rate", f"{latest_rr:.2f}%")
        
        # Explanation
        st.markdown("""
        <div class="highlight">
        <h4>What is Recovery Rate?</h4>
        <p>The Recovery Rate is the proportion of people who have recovered from a disease compared to the total number of people diagnosed with the disease. 
        It is often expressed as a percentage and is calculated as:</p>
        <p>Recovery Rate = (Number of recovered / Number of confirmed cases) × 100</p>
        <p>A higher recovery rate indicates better healthcare outcomes and treatment effectiveness.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Active Case Ratio Analysis")
        active_data = calculate_active_case_ratio(df, selected_country)
        
        # Plot active case ratio over time
        fig = px.line(active_data, x='Date', y='Active_Case_Ratio',
                      title=f"Active Case Ratio Over Time in {selected_country}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest active case ratio
        latest_acr = active_data['Active_Case_Ratio'].iloc[-1]
        st.metric("Latest Active Case Ratio", f"{latest_acr:.2f}%")
        
        # Explanation
        st.markdown("""
        <div class="highlight">
        <h4>What is Active Case Ratio?</h4>
        <p>The Active Case Ratio is the proportion of currently active cases compared to the total number of confirmed cases. 
        It is often expressed as a percentage and is calculated as:</p>
        <p>Active Case Ratio = (Number of active cases / Number of confirmed cases) × 100</p>
        <p>This metric helps understand the current burden of the disease on the healthcare system.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Comparative Analysis")
        
        # Select countries to compare
        st.write("Select countries to compare:")
        col1, col2 = st.columns(2)
        with col1:
            country1 = st.selectbox("Country 1", countries, index=countries.index(selected_country))
        with col2:
            country2 = st.selectbox("Country 2", [c for c in countries if c != country1], 
                                   index=min(countries.index(selected_country) + 1, len(countries) - 1))
        
        # Get data for both countries
        country1_data = df[df['Country/Region'] == country1].groupby('Date').agg({
            'Confirmed': 'sum',
            'Deaths': 'sum',
            'Recovered': 'sum',
            'Active': 'sum'
        }).reset_index()
        
        country2_data = df[df['Country/Region'] == country2].groupby('Date').agg({
            'Confirmed': 'sum',
            'Deaths': 'sum',
            'Recovered': 'sum',
            'Active': 'sum'
        }).reset_index()
        
        # Calculate rates
        country1_mortality = calculate_mortality_rate(df, country1)
        country2_mortality = calculate_mortality_rate(df, country2)
        
        country1_recovery = calculate_recovery_rate(df, country1)
        country2_recovery = calculate_recovery_rate(df, country2)
        
        country1_active = calculate_active_case_ratio(df, country1)
        country2_active = calculate_active_case_ratio(df, country2)
        
        # Latest rates
        country1_latest_cfr = country1_mortality['Case_Fatality_Rate'].iloc[-1]
        country2_latest_cfr = country2_mortality['Case_Fatality_Rate'].iloc[-1]
        
        country1_latest_rr = country1_recovery['Recovery_Rate'].iloc[-1]
        country2_latest_rr = country2_recovery['Recovery_Rate'].iloc[-1]
        
        country1_latest_acr = country1_active['Active_Case_Ratio'].iloc[-1]
        country2_latest_acr = country2_active['Active_Case_Ratio'].iloc[-1]
        
        # Display comparison
        st.write("### Rate Comparison")
        
        # Case Fatality Rate comparison
        fig = go.Figure(data=[
            go.Bar(name=country1, x=['Case Fatality Rate'], y=[country1_latest_cfr]),
            go.Bar(name=country2, x=['Case Fatality Rate'], y=[country2_latest_cfr])
        ])
        fig.update_layout(title="Case Fatality Rate Comparison", yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recovery Rate comparison
        fig = go.Figure(data=[
            go.Bar(name=country1, x=['Recovery Rate'], y=[country1_latest_rr]),
            go.Bar(name=country2, x=['Recovery Rate'], y=[country2_latest_rr])
        ])
        fig.update_layout(title="Recovery Rate Comparison", yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Active Case Ratio comparison
        fig = go.Figure(data=[
            go.Bar(name=country1, x=['Active Case Ratio'], y=[country1_latest_acr]),
            go.Bar(name=country2, x=['Active Case Ratio'], y=[country2_latest_acr])
        ])
        fig.update_layout(title="Active Case Ratio Comparison", yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Confirmed cases comparison
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=country1_data['Date'], y=country1_data['Confirmed'], name=country1))
        fig.add_trace(go.Scatter(x=country2_data['Date'], y=country2_data['Confirmed'], name=country2))
        fig.update_layout(title="Confirmed Cases Comparison", xaxis_title="Date", yaxis_title="Number of Cases")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 