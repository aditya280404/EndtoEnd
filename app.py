import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, redirect
from io import StringIO
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the pre-trained LSTM model
model = load_model('artifacts/lstm_model.h5')

# Load the preprocessor object
with open('artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

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
        if 'file' not in request.files:
            return redirect('/')
        
        file = request.files['file']
        if not file:
            return redirect('/')
        
        # Read CSV data
        csv_data = file.read().decode('utf-8')
        # Parse CSV data
        df = pd.read_csv(StringIO(csv_data))

        # Ensure column names are appropriately handled
        df.columns = df.columns.str.strip()

        # Encode the date column
        label_encoder = LabelEncoder()
        df['date_encoded'] = label_encoder.fit_transform(df['date'])
        
        # Apply MinMax scaling to the '4. close' column
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['4. close_scaled'] = scaler.fit_transform(df[['4. close']])

        # Preprocess only the 'date_encoded' column
        date_encoded_preprocessed = preprocessor.transform(df[['date_encoded']])

        # Convert the preprocessed data back to a DataFrame
        df_preprocessed = pd.DataFrame(date_encoded_preprocessed, columns=['date_encoded'])

        # Keep the '4. close_scaled' column and use it for predictions
        df_preprocessed['4. close_scaled'] = df['4. close_scaled'].values

        # Extract close prices and dates from the preprocessed data
        close_prices_scaled = df_preprocessed['4. close_scaled'].values.reshape(-1, 1)
        dates = df_preprocessed['date_encoded'].values.reshape(-1, 1)

        # Initialize the sequence with the last 60 days of data
        sequence_length = 60
        X = []
        X.append(np.hstack((close_prices_scaled[-sequence_length:], dates[-sequence_length:])))
        X = np.array(X)

        # Now make predictions for the next 30 days
        predictions_scaled = []
        for i in range(30):
            # Predict the next day
            prediction = model.predict(X)[0]
            predictions_scaled.append(prediction)

            # Update the sequence: remove the first entry and append the new prediction
            new_seq = np.vstack((X[0][1:], np.hstack((prediction, dates[-1]))))  # Appending the prediction and the most recent date
            X = np.array([new_seq])

        # Inverse transform the predictions to get them back to the original scale
        predictions = scaler.inverse_transform(predictions_scaled)

        # Pass 'enumerate' to the template
        return render_template('predictions.html', predictions=predictions, enumerate=enumerate)

    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0',port=500)
    