# sales_forecast: Predictive analysis with ARIMA
## Purpose: 
1. Provide an overview time series regression models
2. Model demand for assembly system SKUs based on past historical sales trends

## Overview:
What is time series forecasting?
Time series forecasting is the process of predicting future values of a time series(chronologically indexed data points) by applying statical models
- use cases:
  - predicting and explaining seasonal sales patterns
  - detecting unusual events and calculate their magnitude
  - estimate the effect of newly launched products on number of similar units sold

Components of time forecasting: Trend, Seasonality, Cyclical component, Noise
 
What is ARIMA?
ARIMA is the integration of the autoregressive(AR) model, where previous values have weighted influence on predicted values, and the moving average (MA) models which accounts for noise in predicted values.

Evaluation technique: 
- mean absolute error: shows on average how much forecast differs from actual value
