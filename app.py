import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
import csv
from io import StringIO

# Load the pre-trained LSTM model
model = load_model('artifacts/lstm_model.h5')

# Initialize Flask app
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.files['file']:
            # Get CSV file from the request
            file = request.files['file']
            # Read CSV data
            csv_data = file.read().decode('utf-8')
            # Parse CSV data
            df = pd.read_csv(StringIO(csv_data))

            # Extract close prices and dates from the CSV data
            close_prices = df['4. close'].values.reshape(-1, 1)
            dates = pd.to_datetime(df['date']).values.reshape(-1, 1)

            # Normalize features using Min-Max scaling
            scaler = MinMaxScaler(feature_range=(0, 1))
            close_prices_scaled = scaler.fit_transform(close_prices)
            dates_scaled = scaler.fit_transform(dates)

            # Combine normalized features into sequences
            sequence_length = 60  # Adjusted to match model's expected sequence length
            X = []
            for i in range(len(close_prices_scaled) - sequence_length + 1):
                seq_close = close_prices_scaled[i:i + sequence_length]
                seq_dates = dates_scaled[i:i + sequence_length]
                X.append(np.hstack((seq_close, seq_dates)))

            # Convert X to numpy array
            X = np.array(X)

            # Make predictions
            predictions = model.predict(X)

            # Return the predictions as JSON
            return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({'error': 'No CSV file provided'}), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=8083)
