from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load the trained LSTM model
model = load_model('bitcoin_lstm_model.h5')

# Load the dataset for feature extraction
df = pd.read_csv("bitcoin_2017_to_2023.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.resample("D", on="timestamp").max()  # Resampling the data
df.drop(['volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'], axis=1, inplace=True)

# Reset index to make 'timestamp' a column
df.reset_index(inplace=True)

# Scale the data
scaler = MinMaxScaler()
scaler.fit(df['close'].values.reshape(-1, 1))

window_size = 60

def preprocess_data(date):
    # Log the received date
    logging.info(f"Received date for preprocessing: {date}")

    # Check if the date exists in the dataset
    if date not in df['timestamp'].dt.date.values:
        raise ValueError("Date not found in the dataset.")
    
    # Get the close price history for the last 60 days before the specified date
    recent_data = df.loc[df['timestamp'].dt.date < date].tail(window_size)
    
    if len(recent_data) < window_size:
        raise ValueError("Not enough historical data to make a prediction.")
    
    # Prepare the features
    features = scaler.transform(recent_data['close'].values.reshape(-1, 1))
    return np.reshape(features, (1, window_size, 1))

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.json
    try:
        # Extract the date from the request
        date = pd.to_datetime(data['date']).date()  # Convert to date

        # Preprocess the data
        features = preprocess_data(date)

        # Make a prediction
        prediction = model.predict(features)
        predicted_price = scaler.inverse_transform(prediction)

        # Convert the predicted price to a standard Python float
        predicted_price_value = predicted_price[0][0].item()  # or float(predicted_price[0][0])

        return jsonify({'predicted_close': predicted_price_value})
    except ValueError as ve:
        logging.error(f"Preprocessing error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400  # Return 400 Bad Request for value errors
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        return jsonify({'error': str(e)}), 500  # Return 500 Internal Server Error for other exceptions


if __name__ == "__main__":
    app.run(debug=True)