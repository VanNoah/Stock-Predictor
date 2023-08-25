import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

# Function to prepare stock data
def prepare_stock_data(company, start_date, end_date, price_column="Close", prediction_days=60, test_ratio=0.2, random_split=True):
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
        scaled_data = scaled_data["data"]
    # Downloads data if scaled_data_file doesnt exist
    else:
        #download from yahoo finance and save file
        data = yf.download(company, start=start_date, end=end_date, progress=False)
        data.to_csv(os.path.join(results_folder, f"{company}_data.csv"))

        # Requirement b: Dealing with NaN issues
        # dropping data is the most common resolution of bad data
        data = data.dropna()

        # Requirement e: Scaling feature columns and storing scaler
        # scaled 0,1 unsure if there are better scaling for this data set but 0,1 seems to be "standard"
        scaler = MinMaxScaler(feature_range=(0, 1))
        # transforming the data to the required format
        scaled_data = scaler.fit_transform(data[price_column].values.reshape(-1, 1))

        # Requirement d: Save data to a local file
        # saves scaled data to a pickle file
        pd.to_pickle({"scaler": scaler, "data": scaled_data}, scaled_data_file)
    # Calculate the number of data points that will be used for training the model
    train_size = int(len(scaled_data) * (1 - test_ratio))

    if random_split:
        # Requirement c: Different methods for splitting data
        # Randomly select data to assign data for training and testing at the specified train_size ratio
        train_indices = np.random.choice(len(scaled_data) - prediction_days, train_size, replace=False)
    else:
        # Sequentially select the initial train_size% of indices for training and the remaining for testing
        train_indices = range(train_size)
    # training arrays predicts the next point of sequence x and stored in y
    x_train = []
    y_train = []
    # loops through training data
    for idx in train_indices:
        # creates training sqeuence
        x_train.append(scaled_data[idx:idx + prediction_days])
        # target value for training sequence
        y_train.append(scaled_data[idx + prediction_days])
    #creates numpi arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #reshapes arrays to the expected lstm format.
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train

# comapny wanted.
company_name = "TSLA"
start_date = "2015-01-01"
end_date = "2022-12-31"

# Data preparation
x_train, y_train = prepare_stock_data(company_name, start_date, end_date)
print("Data preparation complete.")
