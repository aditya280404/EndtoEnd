import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "lstm_model.pkl")
    tuner_dir: str = os.path.join("artifacts", "tuner")

class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                       return_sequences=True,
                       input_shape=(60, 2))) 
        model.add(Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=False))
        model.add(Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def create_sequences(self, data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            target = data[i + seq_length, 0]  
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info("Preparing training and test data")

           
           
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)
            
            
            seq_length = 60
            X_train, y_train = self.create_sequences(train_data, seq_length)
            X_test, y_test = self.create_sequences(test_data, seq_length)

            X_train = X_train.reshape((X_train.shape[0], seq_length, X_train.shape[2]))
            X_test = X_test.reshape((X_test.shape[0], seq_length, X_test.shape[2]))

            logging.info("Initializing Keras Tuner for hyperparameter tuning")
            tuner = RandomSearch(
                LSTMHyperModel(),
                objective='val_loss',
                max_trials=5, 
                executions_per_trial=1,
                directory=self.model_trainer_config.tuner_dir,
                project_name='lstm_tuning'
            )

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

            tuner.search(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop, reduce_lr])

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            logging.info(f"Building the model with the best hyperparameters: {best_hps.values}")

            model = tuner.hypermodel.build(best_hps)

            logging.info(f"Training the best model.")
            model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop, reduce_lr])

            logging.info(f"Saving the trained LSTM model.")
            model.save(self.model_trainer_config.trained_model_file_path)

            logging.info(f"Evaluating the model.")
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"R2 Score: {r2}")

            return r2

        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     try:
#         # Example usage
#         # Assume train_data and test_data are numpy arrays obtained from preprocessing
#         train_data = np.array(...)  # Replace with actual train data
#         test_data = np.array(...)  # Replace with actual test data

#         # Model Training
#         model_trainer = ModelTrainer()
#         r2_score = model_trainer.initiate_model_trainer(train_data, test_data)
#         print(f"Model R2 Score: {r2_score}")

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         sys.exit(1)
