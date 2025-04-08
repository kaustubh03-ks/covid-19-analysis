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
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .nav-item {
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-item:hover {
        background-color: #e6e9ef;
    }
    .nav-item.active {
        background-color: #4c78a8;
        color: white;
    }
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1f77b4;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2c3e50;
        border-bottom: 2px solid #e6e9ef;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 500;
        margin: 1.5rem 0 1rem 0;
        color: #34495e;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    .tab-content {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-top: 1rem;
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
    # Custom navigation
    st.markdown('<h1 class="dashboard-title">🦠 COVID-19 Global Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar navigation
    st.sidebar.markdown("## Navigation")
    
    # Create a more visually appealing navigation
    nav_options = ["Global Overview", "Country Analysis", "Epidemiological Analysis"]
    
    # Use session state to track the active page
    if 'active_page' not in st.session_state:
        st.session_state.active_page = "Global Overview"
    
    # Create navigation buttons
    for option in nav_options:
        if st.sidebar.button(option, key=f"nav_{option}", use_container_width=True):
            st.session_state.active_page = option
    
    # Highlight the active page
    st.sidebar.markdown(f"""
        <style>
        [data-testid="stButton"] button[kind="secondary"]:nth-of-type({nav_options.index(st.session_state.active_page) + 1}) {{
            background-color: #4c78a8;
            color: white;
            border: none;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Display the selected page
    if st.session_state.active_page == "Global Overview":
        show_global_overview(df)
    elif st.session_state.active_page == "Country Analysis":
        show_country_analysis(df)
    else:
        show_epidemiological_analysis(df)

def show_global_overview(df):
    st.markdown('<h2 class="section-header">Global Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics with enhanced styling
    st.markdown('<h3 class="subsection-header">Key Metrics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Confirmed Cases</div>
            </div>
        """.format(df['Confirmed'].max()), unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Deaths</div>
            </div>
        """.format(df['Deaths'].max()), unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Recovered</div>
            </div>
        """.format(df['Recovered'].max()), unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Active Cases</div>
            </div>
        """.format(df['Active'].max()), unsafe_allow_html=True)
    
    # Global trends
    st.markdown('<h3 class="subsection-header">Global Trends</h3>', unsafe_allow_html=True)
    global_data = df.groupby('Date').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Recovered': 'sum',
        'Active': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=global_data['Date'], y=global_data['Confirmed'], name='Confirmed', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=global_data['Date'], y=global_data['Deaths'], name='Deaths', line=dict(color='#d62728', width=2)))
    fig.add_trace(go.Scatter(x=global_data['Date'], y=global_data['Recovered'], name='Recovered', line=dict(color='#2ca02c', width=2)))
    fig.update_layout(
        title="Global COVID-19 Cases Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Cases",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional analysis
    st.markdown('<h3 class="subsection-header">Regional Impact</h3>', unsafe_allow_html=True)
    regional_data = df.groupby('WHO Region').agg({
        'Confirmed': 'max',
        'Deaths': 'max',
        'Recovered': 'max',
        'Active': 'max'
    }).reset_index()
    
    fig = px.bar(regional_data, x='WHO Region', y=['Confirmed', 'Deaths', 'Recovered'],
                 title="COVID-19 Impact by WHO Region",
                 barmode='group',
                 color_discrete_sequence=['#1f77b4', '#d62728', '#2ca02c'])
    fig.update_layout(
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Global rates
    st.markdown('<h3 class="subsection-header">Global Rates</h3>', unsafe_allow_html=True)
    mortality_data = calculate_mortality_rate(df)
    recovery_data = calculate_recovery_rate(df)
    active_data = calculate_active_case_ratio(df)
    
    # Latest rates
    latest_mortality = mortality_data['Case_Fatality_Rate'].iloc[-1]
    latest_recovery = recovery_data['Recovery_Rate'].iloc[-1]
    latest_active = active_data['Active_Case_Ratio'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}%</div>
                <div class="metric-label">Global Case Fatality Rate</div>
            </div>
        """.format(latest_mortality), unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}%</div>
                <div class="metric-label">Global Recovery Rate</div>
            </div>
        """.format(latest_recovery), unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}%</div>
                <div class="metric-label">Global Active Case Ratio</div>
            </div>
        """.format(latest_active), unsafe_allow_html=True)

def show_country_analysis(df):
    st.markdown('<h2 class="section-header">Country Analysis</h2>', unsafe_allow_html=True)
    
    # Country selector with enhanced styling
    st.markdown('<h3 class="subsection-header">Select Country</h3>', unsafe_allow_html=True)
    countries = sorted(df['Country/Region'].unique())
    selected_country = st.selectbox("", countries, index=countries.index("US") if "US" in countries else 0)
    
    # Country-specific metrics
    country_data = df[df['Country/Region'] == selected_country]
    latest_data = country_data.loc[country_data['Date'] == country_data['Date'].max()]
    
    st.markdown('<h3 class="subsection-header">Key Metrics</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Confirmed Cases</div>
            </div>
        """.format(latest_data['Confirmed'].sum()), unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Deaths</div>
            </div>
        """.format(latest_data['Deaths'].sum()), unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Recovered</div>
            </div>
        """.format(latest_data['Recovered'].sum()), unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Active Cases</div>
            </div>
        """.format(latest_data['Active'].sum()), unsafe_allow_html=True)
    
    # Country trends
    st.markdown(f'<h3 class="subsection-header">Trends for {selected_country}</h3>', unsafe_allow_html=True)
    country_trends = country_data.groupby('Date').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Recovered': 'sum',
        'Active': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=country_trends['Date'], y=country_trends['Confirmed'], name='Confirmed', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=country_trends['Date'], y=country_trends['Deaths'], name='Deaths', line=dict(color='#d62728', width=2)))
    fig.add_trace(go.Scatter(x=country_trends['Date'], y=country_trends['Recovered'], name='Recovered', line=dict(color='#2ca02c', width=2)))
    fig.update_layout(
        title=f"COVID-19 Cases in {selected_country}",
        xaxis_title="Date",
        yaxis_title="Number of Cases",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth rate analysis
    st.markdown('<h3 class="subsection-header">Daily Growth Rate</h3>', unsafe_allow_html=True)
    growth_data = calculate_growth_rate(df, selected_country)
    fig = px.line(growth_data, x='Date', y='Growth_Rate',
                  title=f"Daily Growth Rate of Confirmed Cases in {selected_country}",
                  color_discrete_sequence=['#ff7f0e'])
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Country-specific rates
    st.markdown(f'<h3 class="subsection-header">Rates for {selected_country}</h3>', unsafe_allow_html=True)
    mortality_data = calculate_mortality_rate(df, selected_country)
    recovery_data = calculate_recovery_rate(df, selected_country)
    active_data = calculate_active_case_ratio(df, selected_country)
    
    # Latest rates
    latest_mortality = mortality_data['Case_Fatality_Rate'].iloc[-1]
    latest_recovery = recovery_data['Recovery_Rate'].iloc[-1]
    latest_active = active_data['Active_Case_Ratio'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}%</div>
                <div class="metric-label">Case Fatality Rate</div>
            </div>
        """.format(latest_mortality), unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}%</div>
                <div class="metric-label">Recovery Rate</div>
            </div>
        """.format(latest_recovery), unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}%</div>
                <div class="metric-label">Active Case Ratio</div>
            </div>
        """.format(latest_active), unsafe_allow_html=True)

def show_epidemiological_analysis(df):
    st.markdown('<h2 class="section-header">Epidemiological Analysis</h2>', unsafe_allow_html=True)
    
    # Country selector
    st.markdown('<h3 class="subsection-header">Select Country</h3>', unsafe_allow_html=True)
    countries = sorted(df['Country/Region'].unique())
    selected_country = st.selectbox("", countries, index=countries.index("US") if "US" in countries else 0)
    
    # Tabs for different analyses with enhanced styling
    tab1, tab2, tab3, tab4 = st.tabs(["Case Fatality Analysis", "Recovery Analysis", "Active Case Analysis", "Comparative Analysis"])
    
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown(f'<h3 class="subsection-header">Case Fatality Rate Analysis for {selected_country}</h3>', unsafe_allow_html=True)
        mortality_data = calculate_mortality_rate(df, selected_country)
        
        # Plot case fatality rate over time
        fig = px.line(mortality_data, x='Date', y='Case_Fatality_Rate',
                      title=f"Case Fatality Rate Over Time in {selected_country}",
                      color_discrete_sequence=['#d62728'])
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest case fatality rate
        latest_cfr = mortality_data['Case_Fatality_Rate'].iloc[-1]
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}%</div>
                <div class="metric-label">Latest Case Fatality Rate</div>
            </div>
        """.format(latest_cfr), unsafe_allow_html=True)
        
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown(f'<h3 class="subsection-header">Recovery Rate Analysis for {selected_country}</h3>', unsafe_allow_html=True)
        recovery_data = calculate_recovery_rate(df, selected_country)
        
        # Plot recovery rate over time
        fig = px.line(recovery_data, x='Date', y='Recovery_Rate',
                      title=f"Recovery Rate Over Time in {selected_country}",
                      color_discrete_sequence=['#2ca02c'])
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest recovery rate
        latest_rr = recovery_data['Recovery_Rate'].iloc[-1]
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}%</div>
                <div class="metric-label">Latest Recovery Rate</div>
            </div>
        """.format(latest_rr), unsafe_allow_html=True)
        
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown(f'<h3 class="subsection-header">Active Case Ratio Analysis for {selected_country}</h3>', unsafe_allow_html=True)
        active_data = calculate_active_case_ratio(df, selected_country)
        
        # Plot active case ratio over time
        fig = px.line(active_data, x='Date', y='Active_Case_Ratio',
                      title=f"Active Case Ratio Over Time in {selected_country}",
                      color_discrete_sequence=['#9467bd'])
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest active case ratio
        latest_acr = active_data['Active_Case_Ratio'].iloc[-1]
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}%</div>
                <div class="metric-label">Latest Active Case Ratio</div>
            </div>
        """.format(latest_acr), unsafe_allow_html=True)
        
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown('<h3 class="subsection-header">Comparative Analysis</h3>', unsafe_allow_html=True)
        
        # Select countries to compare
        st.markdown('<h4>Select countries to compare:</h4>', unsafe_allow_html=True)
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
        st.markdown('<h4>Rate Comparison</h4>', unsafe_allow_html=True)
        
        # Case Fatality Rate comparison
        fig = go.Figure(data=[
            go.Bar(name=country1, x=['Case Fatality Rate'], y=[country1_latest_cfr], marker_color='#1f77b4'),
            go.Bar(name=country2, x=['Case Fatality Rate'], y=[country2_latest_cfr], marker_color='#d62728')
        ])
        fig.update_layout(
            title="Case Fatality Rate Comparison",
            yaxis_title="Percentage (%)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recovery Rate comparison
        fig = go.Figure(data=[
            go.Bar(name=country1, x=['Recovery Rate'], y=[country1_latest_rr], marker_color='#1f77b4'),
            go.Bar(name=country2, x=['Recovery Rate'], y=[country2_latest_rr], marker_color='#d62728')
        ])
        fig.update_layout(
            title="Recovery Rate Comparison",
            yaxis_title="Percentage (%)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Active Case Ratio comparison
        fig = go.Figure(data=[
            go.Bar(name=country1, x=['Active Case Ratio'], y=[country1_latest_acr], marker_color='#1f77b4'),
            go.Bar(name=country2, x=['Active Case Ratio'], y=[country2_latest_acr], marker_color='#d62728')
        ])
        fig.update_layout(
            title="Active Case Ratio Comparison",
            yaxis_title="Percentage (%)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confirmed cases comparison
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=country1_data['Date'], y=country1_data['Confirmed'], name=country1, line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=country2_data['Date'], y=country2_data['Confirmed'], name=country2, line=dict(color='#d62728', width=2)))
        fig.update_layout(
            title="Confirmed Cases Comparison",
            xaxis_title="Date",
            yaxis_title="Number of Cases",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 