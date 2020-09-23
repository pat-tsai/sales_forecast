# importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima as AA
import warnings
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import regex as re

# error evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# suppress harmless deprecation warnings
warnings.filterwarnings("ignore")

# importing dataset
system_forecasts = pd.read_csv('datasets/systems_copied.csv', parse_dates=True, index_col=0)

transposed = system_forecasts.transpose()
transposed.replace(',','', regex=True, inplace=True)    # removing all commas from dataframe
transposed.index = pd.to_datetime([re.sub(r'[US]*[US]*',r'', value) for value in transposed.index.values])

# transforming dataset (dates reversed)
reversed_transposed = transposed.iloc[::-1]


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
    for i in range(len(header_row)):
        ETS_decomp = seasonal_decompose(cleaned_transposed[header_row[i]], model='additive')
        ETS_decomp.plot()
        my_file = (f'{header_row[i]}_ETS_decomp_visualization.png')  # naming visualization
        plt.savefig(os.path.join(my_path, my_file))
        plt.clf()

def get_ARIMA_visuals(header_row,i):
    curr_dir = os.getcwd()
    my_path = curr_dir + '/ARIMA_predictions'
    my_file = (f'{header_row[i]}_forecast.png')  # naming visualization
    plt.savefig(os.path.join(my_path, my_file))
    plt.clf()

DataFrameDict = {sku: pd.DataFrame for sku in header_row}
for key in DataFrameDict.keys():
    DataFrameDict[key] = cleaned_transposed[key]


def forecast(dataframe, model, sku_name):
    curr_dir = os.getcwd()
    my_path = curr_dir + '/ARIMA_forecasts'

    # Forecast for next 1 year
    forecast = model.predict(start=len(dataframe),
                              end=(len(dataframe) - 1) + 1 * 12,
                              typ='levels').rename('Forecast')

    # Plot the forecast values
    dataframe.plot(figsize=(12, 5), legend=True)
    forecast.plot(legend=True)
    my_file = (f'{sku_name}_prediction.png')  # naming visualization
    plt.savefig(os.path.join(my_path, my_file))
    plt.clf()


# splits data into training and test datasets, outputs forecasted values
def train_test_predict():
    curr_dir = os.getcwd()
    my_path = curr_dir + '/ARIMA_predictions'
    error_dict = {}

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
        train = df.iloc[:len(df) - 12]  # allocate all but 12 mo data for training
        test = df.iloc[len(df) - 12:]  # 12 mo data for testing

        try:    # try producing predicted values
            model = SARIMAX(train,
                            order=stepwise_fit.get_params()['order'],
                            seasonal_order=stepwise_fit.get_params()['seasonal_order'])

            result = model.fit()
            forecast(df, result, header)    # calculating 1 year forecast

            # calculating predicted values
            start = len(train)
            end = len(train) + len(test) - 1

            # Predictions for one-year against the test set
            predictions = result.predict(start, end, typ='levels').rename("Predictions")

            # graphing actual vs predicted values
            predictions.plot(figsize=(12,5), label='Predicted #')
            test.plot(label='Actual #')
            plt.legend()
            plt.ylabel('Quantity')
            plt.xlabel('Month')
            plt.title(f'{header} predicted vs actual demand')
            #get_ARIMA_visuals(header_row, i)

            my_file = (f'{header}_prediction.png')  # naming visualization
            plt.savefig(os.path.join(my_path, my_file))
            plt.clf()

            # generating error report
            error_dict[header] = [rmse(test, predictions),mean_squared_error(test, predictions)]
        except:
            print(f'Analysis for {header} failed.. skipping')
            error_dict[header] = ['null','null']
            pass

    error_log = pd.DataFrame.from_dict(error_dict, orient='index', columns=['RMSE','MSE'])
    error_log.to_csv('prediction_error_log.csv')

#print(train_test_predict())

if __name__ == '__main__':
    get_ETS_visuals()
    train_test_predict()

