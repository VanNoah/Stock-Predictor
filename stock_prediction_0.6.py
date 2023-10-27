# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, SimpleRNN, GRU
from statsmodels.tsa.arima.model import ARIMA

#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
def prepare_stock_data(company, start_date, end_date, price_column, prediction_days=60, test_ratio=0.2, random_split=True):
    # Requirement a: Specifying the start and end dates
    results_folder = "results"
    # Checks if folder exists and makes one if it isnt
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # Directory of data file
    scaled_data_file = os.path.join(results_folder, f"{company}_scaled_data.pkl")
    # If file exists import it
    if os.path.exists(scaled_data_file):
        # Requirement d: Load data from a local file if it exists
        scaled_data = pd.read_pickle(scaled_data_file)
        scaler = scaled_data["scaler"]
        data = scaled_data["data"]
    # Downloads data if scaled_data_file doesnt exist
    else:
        #download from yahoo finance and save file
        data = yf.download(company, start=start_date, end=end_date, progress=False)
        data = data.dropna()
        data.to_csv(os.path.join(results_folder, f"{company}_data.csv"))

        # Requirement b: Dealing with NaN issues
        # dropping data is the most common resolution of bad data


        # Requirement e: Scaling feature columns and storing scaler
        # scaled 0,1 unsure if there are better scaling for this data set but 0,1 seems to be "standard"
        scaler = MinMaxScaler(feature_range=(0, 1))
        # transforming the data to the required format
        for column in price_column:
            data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))

        # Requirement d: Save data to a local file
        # saves scaled data to a pickle file
        pd.to_pickle({"scaler": scaler, "data": data}, scaled_data_file)
    # Calculate the number of data points that will be used for training the model
    train_size = int(len(data) * (1 - test_ratio))

    if random_split:
        # Requirement c: Different methods for splitting data
        # Randomly select data to assign data for training and testing at the specified train_size ratio
        train_indices = np.random.choice(len(data) - prediction_days, train_size, replace=False)
    else:
        # Sequentially select the initial train_size% of indices for training and the remaining for testing
        train_indices = range(train_size)
    # training arrays predicts the next point of sequence x and stored in y
    x_train = []
    y_train = []
    # loops through training data
    for column in price_column:
        for idx in train_indices:
            # creates training sqeuence
            x_train.append(data[column][idx:idx + prediction_days])
            # target value for training sequence
            y_train.append(data[column][idx + prediction_days])
    #creates numpi arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #reshapes arrays to the expected lstm format.
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    data = pd.read_csv(os.path.join(results_folder, f"{company}_data.csv"))

    return x_train, y_train, data, scaler

def plot_candlestick(company_name, start_date, end_date, n=1):
    # Import data
    data_file = os.path.join("results", f"{company_name}_data.csv")
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Filter data based on date range
    dt_range = pd.date_range(start=start_date, end=end_date)
    data = data[data.index.isin(dt_range)]

    # Resample the data to represent 'n' trading days per candlestick
    data_resampled = data.resample(f'{n}D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Create candlestick chart
    mpf.plot(data_resampled, type='candle', style='yahoo', title=f'Candlestick Chart for {company_name}',
             ylabel='Price', volume=True)
    print("showing " +str(n)+" days per candle stick")

#box plot
def plot_boxplot(company_name, start_date, end_date, n=1):
    # Import data
    data_file = os.path.join("results", f"{company_name}_data.csv")
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Filter data based on date range
    dt_range = pd.date_range(start=start_date, end=end_date)
    data = data[data.index.isin(dt_range)]

    data_resampled = data.resample(f'{n}D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Create a box plot
    data_resampled.boxplot(column=['Open', 'High', 'Low', 'Close'], grid=False)

    # Customize plot labels and title
    plt.xlabel('Box Plot for Closing Price')
    plt.ylabel('Closing Price')
    plt.title(f'Box Plot for {company_name}')

    # Show the box plot
    plt.show()
    print("plotted " +str(n)+ " days of data")

def create_model(x_train, y_train, epochs, batch_size, units=256, cell=LSTM, n_layers=2):
    model = Sequential()
    for _ in range(n_layers):
        model.add(cell(units=units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))

    model.add(cell(units=units))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model


def evaluate_order(p, d, q, train_data):
    try:
        print(f"testing: {p} {d} {q}")
        arima_model = ARIMA(train_data, order=(p, d, q))
        arima_model_fit = arima_model.fit()
        aic = arima_model_fit.aic
        return p, d, q, aic
    except Exception as e:
        return p, d, q, float("inf")

def find_best_arima_order(train_data, p_values, d_values, q_values):
    best_aic = float("inf")
    best_order = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                p, d, q, aic = evaluate_order(p, d, q, train_data)
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    print(f"best order: {best_order}")

    if best_order is None:
        best_order = (1, 0, 1)

    return best_order

def ensemble_arima_lstm(train_data, test_data, columns, p_values, d_values, q_values, lstm_prediction):
    arima_predictions = []

    for column in columns:
        if column == "Close":
            selected_variable = train_data[:, columns.index(column)]
            best_order = find_best_arima_order(selected_variable, p_values, d_values, q_values) #comment 3 lines to not run optimiser
            print(f"Best order is: {best_order}") #comment 3 lines to not run optimiser
            arima_model = ARIMA(selected_variable, order=best_order) #comment 3 lines to not run optimiser
            # arima_model = ARIMA(selected_variable, order=[1,0,1]) # un comment line to not run optimiser
            arima_model_fit = arima_model.fit()
            arima_preds = arima_model_fit.forecast(steps=len(test_data))
            arima_predictions.append(list(arima_preds))

    print("model made")

    ensemble_predictions = np.array(arima_predictions)
    ensemble_predictions = np.transpose(ensemble_predictions)

    # Combine LSTM and ARIMA predictions by taking the average element-wise
    final_ensemble = (lstm_prediction + ensemble_predictions) / 2


    return final_ensemble, ensemble_predictions

COMPANY = "TSLA"
TRAIN_START = '2015-01-01'
TRAIN_END = '2020-01-01'
PRICE_VALUE_TRAIN = ["Close", "Open", "Volume", "High", "Low"]
PREDICTION_DAYS = 60

x_train, y_train, data, scaler = prepare_stock_data(COMPANY, TRAIN_START, TRAIN_END, PRICE_VALUE_TRAIN)

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
epochs = 1
batch_size = 25
units = 256
cell = GRU
layers = 2
model = create_model(x_train, y_train, epochs, batch_size, units, cell, layers)

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2020-01-02'
TEST_END = '2022-12-31'
PRICE_VALUE = "Close"
test_data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)

# The above bug is the reason for the following line of code
test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we
# can use part of it for training and the rest for testing. You need to
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
#arima

arima_x = x_train[:, 0]
p, d, q = list(range(5)), list(range(2)), list(range(5))
arima_prediction, raw_arima = ensemble_arima_lstm(train_data=x_train, test_data=x_test, columns=PRICE_VALUE_TRAIN, p_values=p, d_values=d, q_values=q, lstm_prediction=predicted_prices)
#inverse prices
predicted_prices = scaler.inverse_transform(predicted_prices)
arima_prediction = scaler.inverse_transform(arima_prediction)
raw_arima = scaler.inverse_transform((raw_arima))
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation
#-----------------------------------------------------------------------------


plt.plot(actual_prices, label='Actual')
plt.plot(arima_prediction, label='Predicted')
plt.plot(raw_arima, label='Arimia')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual vs. Predicted W/Arima')
plt.show()
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------

k=5

for x in range(k):
    real_data = [model_inputs[(len(model_inputs) - PREDICTION_DAYS+x):, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??
