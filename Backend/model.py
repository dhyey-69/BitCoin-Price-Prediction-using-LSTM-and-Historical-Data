import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Dropout, LSTM
import matplotlib.pyplot as plt

def train_model():
    # Load the dataset
    # Data set : - https://www.kaggle.com/datasets/jkraak/bitcoin-price-dataset
    df = pd.read_csv("bitcoin_2017_to_2023.csv")
    df['date'] = pd.to_datetime(df['timestamp'])
    df = df.resample("D", on="date").max()  # Resampling the data

    # Drop unnecessary columns
    df.drop(['volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'], axis=1, inplace=True)

    # Prepare data for training
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Ensure the data is in float format
    NumCols = df.columns.drop(['timestamp'])
    df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
    df[NumCols] = df[NumCols].astype('float64')

    # Set test size for later splitting
    test_size = df[df.timestamp.dt.year == 2022].shape[0]

    # Scale the close prices
    scaler = MinMaxScaler()
    scaler.fit(df.close.values.reshape(-1, 1))

    window_size = 60
    train_data = df.close[:-test_size].values
    train_data = scaler.transform(train_data.reshape(-1, 1))

    # Prepare training data
    X_train, y_train = [], []
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Prepare test data
    test_data = df.close[-test_size-60:].values
    test_data = scaler.transform(test_data.reshape(-1, 1))
    X_test, y_test = [], []

    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-window_size:i, 0])
        y_test.append(test_data[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model
    model = define_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=1)

    # Evaluate the model
    result = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    Accuracy = 1 - MAPE

    print("Test Loss:", result)
    print("Test MAPE:", MAPE)
    print("Test Accuracy:", Accuracy)

    # Save the model
    model.save('bitcoin_lstm_model.h5')

    # Optionally, plot results
    plot_results(df, test_size, y_test, y_pred, scaler)

def define_model(window_size):
    input1 = Input(shape=(window_size, 1))
    x = LSTM(units=64, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)  # Changed activation to 'relu'
    dnn_output = Dense(1)(x)

    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Adam')
    model.summary()

    return model

def plot_results(df, test_size, y_test, y_pred, scaler):
    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred = scaler.inverse_transform(y_pred)

    plt.figure(figsize=(15, 6), dpi=150)
    plt.plot(df['timestamp'].iloc[:-test_size], scaler.inverse_transform(df.close[:-test_size].values.reshape(-1, 1)), color='black', lw=2)
    plt.plot(df['timestamp'].iloc[-test_size:], y_test_true, color='blue', lw=2)
    plt.plot(df['timestamp'].iloc[-test_size:], y_test_pred, color='red', lw=2)
    plt.title('Model Performance on Bitcoin Price Prediction', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.legend(['Training Data', 'Actual Test Data', 'Predicted Test Data'], loc='upper left', prop={'size': 15})
    plt.grid(color='white')
    plt.show()

if __name__ == "__main__":
    train_model()
