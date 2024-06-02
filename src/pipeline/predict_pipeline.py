import sys
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "lstm_model.h5")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            # Load the pre-trained LSTM model
            model = load_model(self.model_path)

            # Ensure that '4. close' and 'date' columns are present in features
            required_columns = ['4. close', 'date']
            if not all(col in features.columns for col in required_columns):
                raise ValueError("Required columns ('4. close' and 'date') not found in the DataFrame.")

            # Make predictions directly on features
            predictions = model.predict(features)

            # Return predictions
            return predictions
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, date: str, close_price: float):
        self.date = date
        self.close_price = close_price

    def get_data_as_data_frame(self):
        try:
            # Check if the date string is already in the correct format
            if '-' not in self.date:
                raise ValueError("Date string format is not recognized.")

            # Convert the date string to a datetime object
            date_time_obj = pd.to_datetime(self.date)

            # Convert the datetime object back to a string with the desired format
            formatted_date = date_time_obj.strftime('%Y-%m-%d')

            # Create a DataFrame with the correct dtype for the "date" column
            custom_data_input_dict = {
                "date": [formatted_date],
                "4. close": [self.close_price]
            }

            df = pd.DataFrame(custom_data_input_dict)

            return df
        
        except Exception as e:
            raise CustomException(e, sys)


def main():
    # Create a sample CustomData instance
    custom_data = CustomData(date='2024-06-02', close_price=100.0)

    # Convert CustomData to DataFrame
    sample_input = custom_data.get_data_as_data_frame()

    # Initialize predictor
    predictor = PredictPipeline()

    # Make predictions
    predictions = predictor.predict(sample_input)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
