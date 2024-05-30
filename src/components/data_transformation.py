import sys
import os
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # ColumnTransformer to apply transformations
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, ["date_encoded"]),  # Apply to encoded date column
                ],
                remainder='passthrough'  # Leave other columns untouched
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "4. close"  # Update with your target column name

            # Ensure column names are appropriately handled
            train_df.columns = train_df.columns.str.strip()  
            test_df.columns = test_df.columns.str.strip()

            # Combine train and test data for fitting the LabelEncoder
            combined_dates = pd.concat([train_df['date'], test_df['date']]).unique()

            # Initialize and fit LabelEncoder on the combined date data
            label_encoder = LabelEncoder()
            label_encoder.fit(combined_dates)

            # Encode the date column using the fitted LabelEncoder
            train_df['date_encoded'] = label_encoder.transform(train_df['date'])
            test_df['date_encoded'] = label_encoder.transform(test_df['date'])

            # Preprocess train data
            input_feature_train_df = train_df.drop(columns=[target_column_name, 'date'], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Preprocess test data
            input_feature_test_df = test_df.drop(columns=[target_column_name, 'date'], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Shapes of input feature and target DataFrames:")
            logging.info(f"Train input features shape: {input_feature_train_df.shape}")
            logging.info(f"Train target features shape: {target_feature_train_df.shape}")
            logging.info(f"Test input features shape: {input_feature_test_df.shape}")
            logging.info(f"Test target features shape: {target_feature_test_df.shape}")

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            if input_feature_train_df.empty or input_feature_test_df.empty:
                raise ValueError("Input data is empty. Please check the input data.")
            logging.info("No empty data")

            # print(input_feature_train_df.head(2))
            # print(input_feature_test_df.head(2))

            # Fit and transform train data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # print("Transformed train data:")
            # print(input_feature_train_arr)

            # Transform test data
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            # print("Transformed test data:")
            # print(input_feature_test_arr)

            logging.info("Shapes of transformed input feature arrays:")
            logging.info(f"Train input features shape: {input_feature_train_arr.shape}")
            logging.info(f"Test input features shape: {input_feature_test_arr.shape}")

            # Combine input features with target features
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


