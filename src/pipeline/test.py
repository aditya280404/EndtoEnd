import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the pre-trained LSTM model
model = load_model('artifacts/lstm_model.h5')
print(model.summary())

# Load test data from CSV file
try:
    df = pd.read_csv("artifacts/test.csv")
except FileNotFoundError:
    print("Error: File 'test.csv' not found.")
    exit()

# Extract close price and date columns
try:
    close_prices = df['4. close'].values.reshape(-1, 1)  # Reshape to a single column
    dates = pd.to_datetime(df['date']).astype(int).values.reshape(-1, 1)  # Reshape to a single column
except KeyError:
    print("Error: Required columns ('4. close' and 'date') not found in the CSV file.")
    exit()

# Normalize features using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)
dates_scaled = scaler.fit_transform(dates)

# Combine normalized features into sequences
sequence_length = 5
X = []
for i in range(len(close_prices_scaled) - sequence_length + 1):
    seq_close = close_prices_scaled[i:i + sequence_length]
    seq_dates = dates_scaled[i:i + sequence_length]
    X.append(np.hstack((seq_close, seq_dates)))

# Convert X to numpy array
X = np.array(X)

# Make predictions
try:
    predictions = model.predict(X)
except Exception as e:
    print("Error:", e)
    exit()

# Print the predictions
print("Predictions:")
for i, prediction in enumerate(predictions):
    print(f"Sequence {i+1}: Predicted next close price: {prediction[0]}")
