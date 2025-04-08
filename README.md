# COVID-19 Data Analysis Dashboard

A modern, interactive dashboard for analyzing COVID-19 data using Streamlit. This application provides comprehensive insights into the global COVID-19 pandemic through various visualizations and analytical tools.

## Features

### 1. Global Overview
- Key metrics dashboard
- Global trends visualization
- Regional impact analysis
- Interactive charts and graphs
- Global epidemiological rates

### 2. Country Analysis
- Country-specific metrics
- Trend analysis by country
- Daily growth rate visualization
- Country-specific epidemiological rates

### 3. Epidemiological Analysis
- Case Fatality Rate analysis
- Recovery Rate analysis
- Active Case Ratio analysis
- Comparative analysis between countries
- Educational explanations of key metrics

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

## Data Description

The dataset (`covid_19_clean_complete.csv`) contains COVID-19 data with the following columns:
- Province/State
- Country/Region
- Latitude and Longitude
- Date
- Confirmed cases
- Deaths
- Recovered cases
- Active cases
- WHO Region

## Key Metrics Explained

### Case Fatality Rate (CFR)
The proportion of deaths from COVID-19 compared to the total number of confirmed cases. Calculated as:
```
CFR = (Number of deaths / Number of confirmed cases) × 100
```

### Recovery Rate
The proportion of people who have recovered from COVID-19 compared to the total number of confirmed cases. Calculated as:
```
Recovery Rate = (Number of recovered / Number of confirmed cases) × 100
```

### Active Case Ratio
The proportion of currently active cases compared to the total number of confirmed cases. Calculated as:
```
Active Case Ratio = (Number of active cases / Number of confirmed cases) × 100
```

## Technologies Used

- Streamlit: For creating the interactive web application
- Plotly: For interactive visualizations
- Pandas: For data manipulation and analysis
- Scikit-learn: For forecasting models
- Statsmodels: For statistical analysis

## Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
└── covid_19_clean_complete.csv  # Dataset
```

## Usage

1. Navigate through different sections using the sidebar
2. Select countries for detailed analysis
3. Explore epidemiological metrics in the dedicated section
4. Compare countries using the comparative analysis tab
5. Learn about key COVID-19 metrics through educational explanations

## Future Enhancements

1. Add more epidemiological metrics
2. Implement demographic analysis
3. Add more interactive features
4. Include vaccination data analysis
5. Add export functionality for reports 