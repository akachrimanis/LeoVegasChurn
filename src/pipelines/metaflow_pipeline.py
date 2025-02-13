import os
import sys
import logging

import numpy as np

from metaflow import FlowSpec, step, IncludeFile
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    mean_squared_error,
)

# Add the project root to sys.path
project_root = os.path.abspath(
    "/Users/tkax/dev/aimonetize/WIP/DistroEnergyML"
)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.FE.pandasFE.timeseriesFE import timeseries_components

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format for log messages
)


class MLFlowPipeline(FlowSpec):

    @step
    def start(self):
        # ETL Step: Load data
        self.data = pd.read_csv('/Users/tkax/dev/aimonetize/WIP/DistroEnergyML/data/raw/ami_filtered_data.csv')
        logging.info(f"Data head:\n{self.data.head()}")
        
        self.next(self.data_prep)


    @step
    def data_prep(self):
        # Data preparation: Clean and preprocess data
        self.data.fillna(self.data.kWh.mean(), inplace=True)
        self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])
        #print(self.data.columns)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        # Feature Engineering: Create or transform features
        self.data = self.data.sort_values('Datetime')
        self.data = timeseries_components(self.data, 'Datetime')
        self.data["lag_1"] = self.data.kWh.shift(1)
        self.data = self.data.iloc[1:] # Drop the first row with NaN values
        print(self.data.dtypes)
        for col in self.data.select_dtypes(include='int64').columns:
            self.data[col] = self.data[col].astype(float)
            print("Integer column{col}")
        self.next(self.model_training)

    @step
    def model_training(self):
        # Model Training
        X = self.data.drop(['kWh', 'Datetime'], axis=1)
        print(X.columns)
        y = self.data['kWh']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Prediction and evaluation
        y_pred = self.model.predict(X_test)
  # Calculating metrics on the test set
        best_mse = mean_squared_error(y_test, y_pred)
        best_mae = mean_absolute_error(y_test, y_pred)
        best_r2 = r2_score(y_test, y_pred)
        best_rmse = np.sqrt(best_mse)  # Square root of MSE for RMSE
        self.next(self.mlflow_log)

    @step
    def mlflow_log(self):
        # Logging to MLflow
        mlflow.start_run()
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("best_mse", self.best_mse)
        mlflow.sklearn.log_model(self.model, "model")

        # Tagging and registering the model
        mlflow.set_tag('stage', 'development')
        mlflow.register_model("model", "RandomForestModel")

        mlflow.end_run()
        self.next(self.end)

    @step
    def end(self):
        print("Pipeline completed with an accuracy of", self.accuracy)

if __name__ == '__main__':
    MLFlowPipeline()
