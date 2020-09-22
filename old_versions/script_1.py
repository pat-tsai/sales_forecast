# importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima as AA
import warnings

# suppress harmless deprecation warnings
warnings.filterwarnings("ignore")

# importing dataset
system_forecasts = pd.read_csv('../datasets/systems_SKUs.csv', parse_dates=True, index_col=0)
print(system_forecasts.head())
print(system_forecasts.columns)

# transforming dataset
dimensions=system_forecasts.shape   # dimensiom of dataframe: 10 rows x 13 columns
print(dimensions[1])
#print(system_forecasts.iloc[dimensions[0]])

#first_col = system_forecasts.iloc[:,1].values.tolist() # gets the first column from df and stores all ints in list
header_row = system_forecasts.columns.values.tolist()    # finding header dates
#header_row = header_row[::-1]   # reversing dates to reorder in increasing time

indices = system_forecasts.index.values.tolist()

#print(f'first column is {first_col}')
#print(indices[0])
#print(indices[1])
#print(header_row)
print(type(header_row))
print(indices)

originalColumns = []
# loop through all columns in original dataframe
'''
for i in range(dimensions[1]):
    col = system_forecasts.iloc[:,i].values.tolist()
    if col not in originalColumns:
 #       col=col[::-1]
        originalColumns.append(col)
'''
originalRows = []
# loop through all rows in original df (data values only)
for i in range(dimensions[0]):
    row = system_forecasts.iloc[i].values.tolist()
    if row not in originalColumns:
 #       col=col[::-1]
        originalRows.append(row)
print(originalRows)


print(f'the length of header_row is {len(header_row)}')
print(type(indices[1]))
# reshaped dataframe
date_and_units_sold = {}

#reversed_date_and_units_sold = {}

cleaned_dates = []
for date in header_row:
    date=date.replace('US','')
    cleaned_dates.append(date)

j = 0
for system in indices:
    date_and_units_sold[system] = originalRows[j]
    j += 1

print(date_and_units_sold)
#reversed_date_and_units_sold = {value:key for (key,value) in date_and_units_sold.items()}
#print(reversed_date_and_units_sold)

transformed_system_forecasts = pd.DataFrame(date_and_units_sold, index=cleaned_dates)
print(transformed_system_forecasts)

#pivoted = system_forecasts.pivot(index=header_row,columns=indices,values=[column for column in originalColumns])


# read through each series in above dataframe and call each series on seasonal_decompose
# plot results in subplot

#ETS_decomp = seasonal_decompose()
#stepwise_fit = AA()

# visualizes separate components of a time series (Error, Trend, Seasonality, Residuals)
def find_ETS_decomp():
    my_path = os.path.abspath(__file__)
    my_path = my_path + '/ETS_decomp_visuals'

    for i in  range(len(indices)):
        ETS_decomp = seasonal_decompose(transformed_system_forecasts[indices[i]], model='additive')
        ETS_decomp.plot()
        my_file = (f'{indices[i]} ETS_decomp_visualization.png')   # naming visualization
        plt.savefig(os.path.join(my_path,my_file))
        plt.clf()