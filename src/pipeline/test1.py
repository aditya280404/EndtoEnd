import pickle
import numpy as np
import pandas as pd

# Load the preprocessor object
with open('artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Create a sample DataFrame
sample_data = {
    'date_encoded': np.arange(10),
    '4. close': np.random.rand(10)
}
df = pd.DataFrame(sample_data)

# Preprocess data using the loaded preprocessor
df_preprocessed = preprocessor.transform(df[['date_encoded', '4. close']])

# Inspect the shape of the preprocessed DataFrame
print("Shape of df_preprocessed after transformation:", df_preprocessed.shape)
