import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import StandardScaler
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.train_model import ModelTrainerConfig
from src.components.train_model import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    alpha_vantage_api_key: str = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_API_KEY')  # Update with your API key
    symbol: str = 'AAPL'
    outputsize: str = 'full'  # or 'compact'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def fetch_data_from_alpha_vantage(self):
        logging.info("Fetching the data from Alpha Vantage")
        try:
            ts = TimeSeries(key=self.ingestion_config.alpha_vantage_api_key, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=self.ingestion_config.symbol, outputsize=self.ingestion_config.outputsize)
            data.reset_index(inplace=True)
            logging.info(f"Data columns: {data.columns}")
            logging.info("Data fetched successfully from Alpha Vantage")
            return data
        except Exception as e:
            raise CustomException(e, sys)
    
    def preprocess_data(self, df):
        logging.info("Preprocessing the data")
        try:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df.set_index('date', inplace=True)
            logging.info(f"Preprocessed data columns: {df.columns}")
            # Select the correct column for closing prices
            close_column = [col for col in df.columns if 'close' in col.lower()]
            if not close_column:
                raise CustomException(f"'close' column not found. Available columns: {df.columns}", sys)
            df = df[close_column]
            logging.info("Data preprocessed successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)
    
    def scale_data(self, train_set, test_set):
        logging.info("Scaling the data")
        try:
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_set)
            test_scaled = scaler.transform(test_set)
            logging.info("Data scaling completed")
            return train_scaled, test_scaled, scaler
        except Exception as e:
            raise CustomException(e, sys)
                
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Fetching data
            df = self.fetch_data_from_alpha_vantage()
            df = self.preprocess_data(df)
            logging.info('Data fetched and preprocessed')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=True, header=True)
            
            logging.info("Train-test split initiated")
            split_date = datetime.now() - timedelta(days=365)  # Last year's data for testing
            train_set = df[df.index < split_date]
            test_set = df[df.index >= split_date]
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=True, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=True, header=True)
            
            # Scaling the data

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
   
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    # print(train_arr.shape)
    # print(train_arr)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
