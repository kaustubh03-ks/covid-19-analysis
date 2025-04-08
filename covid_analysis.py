import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

def load_data():
    """Load and preprocess the COVID-19 dataset"""
    df = pd.read_csv('covid_19_clean_complete.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def analyze_global_trends(df):
    """Analyze global COVID-19 trends"""
    global_data = df.groupby('Date').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Recovered': 'sum',
        'Active': 'sum'
    }).reset_index()
    
    return global_data

def plot_global_trends(global_data):
    """Create visualization of global trends"""
    plt.figure(figsize=(12, 6))
    plt.plot(global_data['Date'], global_data['Confirmed'], label='Confirmed')
    plt.plot(global_data['Date'], global_data['Deaths'], label='Deaths')
    plt.plot(global_data['Date'], global_data['Recovered'], label='Recovered')
    plt.title('Global COVID-19 Trends')
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('global_trends.png')
    plt.close()

def analyze_regional_impact(df):
    """Analyze COVID-19 impact by WHO region"""
    regional_data = df.groupby('WHO Region').agg({
        'Confirmed': 'max',
        'Deaths': 'max',
        'Recovered': 'max',
        'Active': 'max'
    }).reset_index()
    
    return regional_data

def plot_regional_impact(regional_data):
    """Create visualization of regional impact"""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=regional_data, x='WHO Region', y='Confirmed')
    plt.title('Total Confirmed Cases by WHO Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('regional_impact.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Analyze global trends
    print("Analyzing global trends...")
    global_data = analyze_global_trends(df)
    plot_global_trends(global_data)
    
    # Analyze regional impact
    print("Analyzing regional impact...")
    regional_data = analyze_regional_impact(df)
    plot_regional_impact(regional_data)
    
    # Print some basic statistics
    print("\nBasic Statistics:")
    print(f"Total number of countries/regions: {df['Country/Region'].nunique()}")
    print(f"Date range: from {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total confirmed cases: {df['Confirmed'].max():,}")
    print(f"Total deaths: {df['Deaths'].max():,}")
    print(f"Total recovered: {df['Recovered'].max():,}")

if __name__ == "__main__":
    main() 