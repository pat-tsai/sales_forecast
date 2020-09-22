# importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima as AA
import warnings
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

# suppress harmless deprecation warnings
warnings.filterwarnings("ignore")

# importing dataset
system_forecasts = pd.read_csv('datasets/systems_copied.csv', parse_dates=True, index_col=0)
print(system_forecasts.head())
print(system_forecasts.columns)

transposed = system_forecasts.transpose()


transposed.index = pd.to_datetime([value.replace('US','') for value in transposed.index.values])
print(transposed.index.dtype)


reversed_transposed = transposed.iloc[::-1]
print(reversed_transposed)

def rename_columns():
    new_cols = {}
    orig_column_names = reversed_transposed.columns.tolist()
    new_column_names = [sku.replace(' ','') for sku in orig_column_names]
    for i in range(len(orig_column_names)):
        new_cols[orig_column_names[i]] = new_column_names[i]
    return new_cols

cleaned_transposed = reversed_transposed.rename(columns=rename_columns())
print(cleaned_transposed)


header_row = cleaned_transposed.columns.tolist()
# visualizes separate components of a time series (Error, Trend, Seasonality, Residuals)
def get_ETS_visuals():
    curr_dir = os.getcwd()
    my_path = curr_dir + '/ETS_decomp_visuals'
    print(my_path)
    for i in range(len(header_row)):
        ETS_decomp = seasonal_decompose(cleaned_transposed[header_row[i]], model='additive')
        ETS_decomp.plot()
        my_file = (f'{header_row[i]}_ETS_decomp_visualization.png')  # naming visualization
        plt.savefig(os.path.join(my_path, my_file))
        plt.clf()

#get_ETS_visuals()


DataFrameDict = {sku: pd.DataFrame for sku in header_row}
for key in DataFrameDict.keys():
    DataFrameDict[key] = cleaned_transposed[key]

print(DataFrameDict['AS-1114S-WN10RT'])

# splits data into training and test datasets, and outputs forecast values
def train_test_predict():
    for header in header_row:

        print(f'-------------------Now analyzing {header} -------------------')
        # call AA to identify optional params and return fitted model
        df = DataFrameDict[header]
        stepwise_fit = AA(df, start_p=1, start_q=1,
                          max_p=3, max_q=3, m=12,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
        print(stepwise_fit.get_params()['seasonal_order'])
        # Split data into train / test sets
        train = df.iloc[:len(df) - 12]
        test = df.iloc[len(df) - 12:]  # set one year(12 months) for testing


        try:
            model = SARIMAX(train,
                            order=stepwise_fit.get_params()['order'],
                            seasonal_order=stepwise_fit.get_params()['seasonal_order'])

            result = model.fit()
            #result.summary()

            start = len(train)
            end = len(train) + len(test) - 1

            # Predictions for one-year against the test set
            predictions = result.predict(start, end, typ='levels').rename("Predictions")

            # graphing actual vs predicted values
            predictions.plot(legend=True)
            test.plot(legend=True)
            plt.show()
            plt.clf()

        except NameError:
            print(f'Non-positive-definite forecast error encountered... skipping')
            pass
        except:
            print(f'Analysis for {header} failed.. skipping')
            pass


print(train_test_predict())
