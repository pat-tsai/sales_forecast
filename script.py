# importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima as AA
import warnings
import os

# suppress harmless deprecation warnings
warnings.filterwarnings("ignore")

# importing dataset
system_forecasts = pd.read_csv('datasets/systems_copied.csv', parse_dates=True, index_col=0)
print(system_forecasts.head())
print(system_forecasts.columns)

# pipelining dataset- transforming for analysis
dimensions=system_forecasts.shape   # dimensiom of dataframe: 10 rows x 13 columns
header_row = system_forecasts.columns.values.tolist()    # finding header dates
indices = system_forecasts.index.values.tolist()

originalRows = []
# loop through all rows in original df (data values only)
for i in range(dimensions[0]):
    row = system_forecasts.iloc[i].values.tolist()
    if row not in originalRows:
        originalRows.append(row)

cleaned_dates = []
for date in header_row:
    date=date.replace('US','')
    cleaned_dates.append(date)

# reshaped dataframe
date_and_units_sold = {}
j = 0
for system in indices:
    date_and_units_sold[system] = originalRows[j]
    j += 1

cleaned_datetimes = pd.DatetimeIndex(data=cleaned_dates)
transformed_system_forecasts = pd.DataFrame(date_and_units_sold, index=cleaned_datetimes)
print(transformed_system_forecasts.head())


# visualizes separate components of a time series (Error, Trend, Seasonality, Residuals)
def get_ETS_visuals():
    curr_dir = os.getcwd()
    my_path = curr_dir + '/ETS_decomp_visuals'
    print(my_path)
    for i in range(len(indices)):
        ETS_decomp = seasonal_decompose(transformed_system_forecasts[indices[i]], model='additive')
        ETS_decomp.plot()
        indices[i].replace(' ', '')
        my_file = (f'{indices[i]}_ETS_decomp_visualization.png')  # naming visualization
        plt.savefig(os.path.join(my_path, my_file))
        plt.clf()
    
#get_ETS_visuals()


# call AA to identify optional params and return fitted model
stepwise_fit1 = AA(transformed_system_forecasts[indices[0]], start_p=1, start_q=1,
                  max_p=3, max_q=3, m=12,
                  start_P=0, seasonal=True,
                  d=None, D=1, trace=True,
                  error_action='ignore',
                  suppress_warnings=True,
                  stepwise=True)

print(stepwise_fit1.summary())
#print(stepwise_fit1.df_model())
print(stepwise_fit1.get_params())

# splits data into training and test datasets, and outputs forecast values
def train_test_predict():

'''
stepwise_fit2 = AA(transformed_system_forecasts[indices[0]], seasonal=True,
                  d=None, D=1, trace=True,
                  error_action='ignore',
                  suppress_warnings=True,
                  stepwise=True)

print(stepwise_fit2.summary())
'''

# splitting data into training and testing sets
train = transformed_system_forecasts
#SARIMAX(1, 0, 0)x(0, 1, 0, 12)