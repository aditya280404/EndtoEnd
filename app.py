import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from src.exception import CustomException
import sys

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
        if 'file' not in request.files:
            return jsonify({'error': 'No CSV file provided'}), 400
        
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No CSV file provided'}), 400

        # Read CSV data
        csv_data = file.read().decode('utf-8')
        # Parse CSV data
        df = pd.read_csv(StringIO(csv_data))

        # Preprocess data
        df, close_price_scaler = preprocess_data(df)

        # Extract close prices and dates from the preprocessed data
        close_prices = df['4. close'].values.reshape(-1, 1)
        dates = df['date_encoded'].values.reshape(-1, 1)

        # Combine normalized features into sequences
        sequence_length = 60  # Adjusted to match model's expected sequence length
        X = []
        for i in range(len(close_prices) - sequence_length + 1):
            seq_close = close_prices[i:i + sequence_length]
            seq_dates = dates[i:i + sequence_length]
            X.append(np.hstack((seq_close, seq_dates)))

        # Convert X to numpy array
        X = np.array(X)

        # Make predictions
        predictions_scaled = model.predict(X)

        # Inverse transform the scaled predictions to original values
        predictions = close_price_scaler.inverse_transform(predictions_scaled)

        # Return the predictions as JSON
        return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def preprocess_data(df):
    try:
        # Ensure column names are appropriately handled
        df.columns = df.columns.str.strip()  

        # Combine train and test data for fitting the LabelEncoder
        combined_dates = pd.concat([df['date']]).unique()

        # Initialize and fit LabelEncoder on the combined date data
        label_encoder = LabelEncoder()
        label_encoder.fit(combined_dates)

        # Encode the date column using the fitted LabelEncoder
        df['date_encoded'] = label_encoder.transform(df['date'])

        # Define the preprocessing pipeline
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, ['date_encoded'])
            ],
            remainder='passthrough'  # Leave other columns untouched
        )

        # Apply the preprocessing pipeline to the dataframe
        df_preprocessed = preprocessor.fit_transform(df[['date_encoded', '4. close']])

        # Convert the preprocessed data back to a DataFrame for easier manipulation
        df_preprocessed = pd.DataFrame(df_preprocessed, columns=['date_encoded', '4. close'])

        # Fit a MinMaxScaler on the close prices for inverse transform later
        close_price_scaler = MinMaxScaler()
        df_preprocessed['4. close'] = close_price_scaler.fit_transform(df_preprocessed[['4. close']])

        return df_preprocessed, close_price_scaler

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=8084)
