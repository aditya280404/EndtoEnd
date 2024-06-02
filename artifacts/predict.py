import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the model
with open('artifacts/lstm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Inspect the model input shape
model_input_shape = model.input_shape
print(f"Expected model input shape: {model_input_shape}")

# Load the data
data = pd.read_csv('artifacts/test.csv', parse_dates=['date'], index_col='date')

# Prepare the data for prediction
# Ensure the data is sorted by date
data = data.sort_index()

# Assuming you need to predict the '4. close' prices
close_data = data[['4. close']]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Create sequences for prediction
def create_sequences(data, n):
    sequences = []
    for i in range(len(data) - n):
        sequences.append(data[i:i+n])
    return np.array(sequences)

sequence_length = 60  # Adjust based on your training
sequences = create_sequences(scaled_data, sequence_length)

# Ensure the shape of the sequences is correct
# The shape should be (number of sequences, sequence_length, number of features)
sequences = np.reshape(sequences, (sequences.shape[0], sequences.shape[1], 1))

# Print the shape of sequences
print(f"Shape of sequences: {sequences.shape}")

# Ensure the sequences match the expected input shape (batch_size, timesteps, features)
# Note: batch_size can be None or a fixed number, so we only need to ensure timesteps and features
if sequences.shape[1:] != model_input_shape[1:]:
    raise ValueError(f"Shape of sequences {sequences.shape} does not match expected model input shape {model_input_shape}")

# Predict the next value (assuming the model was trained to predict the next 'close' value)
predictions = model.predict(sequences)

# Invert scaling
predictions = scaler.inverse_transform(predictions)

# Prepare the dataframe with predictions
pred_dates = data.index[sequence_length:]
pred_df = pd.DataFrame(data=predictions, index=pred_dates, columns=['predicted_close'])

# Print the results
print(pred_df.head())
print(pred_df.tail())