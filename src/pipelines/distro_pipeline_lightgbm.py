
import os
import sys
import pandas as pd
import numpy as np
import yaml
import joblib
import mlflow
import time
import warnings
from datetime import datetime
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor
import importlib
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.base import TransformerMixin, BaseEstimator
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor

# Suppress FutureWarnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the project root to sys.path
project_root = os.path.abspath(
    "/Users/tkax/dev/aimonetize/WIP/DistroEnergyML"
)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.load_config import load_config
from src.FE.normalize import scale_features, select_columns_to_scale
from src.FE.encoding import prepare_column_lists, encode_columns
from src.FE.date_features import create_extended_date_features
from src.FE.pandasFE.timeseriesFE import create_multiple_rolling_features, timeseries_components, timeseries_lags
from src.CV.time_series import perform_time_series_cv
from src.data_quality.quality import report_missing_data
from src.data_prep.change_variable_format import convert_integers_to_float
from src.data_prep.missing_values import impute_selected_columns
from src.EDA.EDA import perform_eda
from src.EDA.dates import identify_date_columns
from src.data_prep.identify_cols import identify_non_numeric_columns_for_model
from src.CV.CV import run_model_with_dynamic_cv

# Patch the XGBRegressor class for compatibility with scikit-learn
def _patched_sklearn_tags(self):
    return {
        "estimator_type": "regressor",
        "requires_y": True,
    }


XGBRegressor.__sklearn_tags__ = _patched_sklearn_tags


def select_model(model_type, model_config):
    """
    Select and return the appropriate model based on the configuration.

    Args:
        - model_type (str): The type of model to select (e.g., "elasticnet", "random_forest", "xgboost").
        - model_config (dict): The configuration dictionary containing model parameters.

    Returns:
        - model: The model instance based on the specified type.
    """
    if model_type == "elasticnet":
        return ElasticNet(**model_config.get("elasticnet", {}))
    elif model_type == "random_forest":
        return RandomForestRegressor(**model_config.get("random_forest", {}))
    elif model_type == "xgboost":
        return XGBRegressor(**model_config.get("xgboost", {}))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def pipeline(config):
    """
    Full pipeline for training and selecting the best model.

    Args:
        - config (dict): The configuration dictionary containing paths, model parameters, etc.
    """

    mlflow.set_tracking_uri(
        config["mlflow"]["set_tracking_uri"]
    )  # Replace with your server URI
    experiment_name = config["mlflow"]["experiment_name"]  # Example: read from config

    def ETL(config):
        """
        Extract, transform, and load (ETL) the data.

        Args:
            - config (dict): The configuration dictionary containing the ETL paths.

        Returns:
            - data (pandas DataFrame): The loaded and processed data.
        """
        raw_data_path = config["etl"]["raw_data_path"]
        processed_data_path = config["etl"]["processed_data_path"]

        data = joblib.load(raw_data_path)
        data = data.head(1000)  # Example: limit to 1000 rows for testing

        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        joblib.dump(data, processed_data_path)
        print(f"Raw data saved to {processed_data_path}.")
        report_missing_data(data)

        return data

    def data_prep(data, config):
        """
        Prepare the data by cleaning and filling missing values.

        Args:
            - data (pandas DataFrame): The raw data to prepare.
            - config (dict): The configuration dictionary.

        Returns:
            - data (pandas DataFrame): The prepared data.
        """
        print("Preparing data...")
        data.set_index('DateTime', inplace=True)
        #data = data.resample('15T').sum()  # '15T' for 15-minute intervals
        print(data.columns)
        print(data.head())
        # Drop duplicate rows
        data = data.drop_duplicates()
        data.loc[:,'meter_id'] = 'meter1'

        # Fill missing values with mean (for simplicity in this example)
        data = convert_integers_to_float(data)
        data = impute_selected_columns(data, strategy="mean")
        # categorical_columns, target_column, categories = prepare_column_lists(df, target_column='target', ordinal_columns=ordinal_columns)
        data = data.reset_index()
        data = timeseries_components(data, datetime_col=config["FE"]["date_column"])

        report_missing_data(data)
       
        return data
    
    def FE(data, config):
        """
        Feature engineering step: encoding categorical variables and scaling.

        Args:
            - data (pandas DataFrame): The prepared data.
            - config (dict): The configuration dictionary containing variables to encode and scale.

        Returns:
            - data (pandas DataFrame): The engineered data with one-hot encoding, scaling, and integer conversion.
        """
        print("Performing feature engineering...")
        data = data.copy()

        data = create_multiple_rolling_features(df=data, column=config["FE"]["metric"], window_sizes=[4,8,16,32,64,128], features=['mean', 'std', 'max', 'min'])        
        data = timeseries_lags(data, column_id='meter_id', column_sort=config["FE"]["date_column"], metric=config["FE"]["metric"], lag_values=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        print(data.head())
        print(data.dtypes)
        report_missing_data(data)
        data = data.drop(columns=config["FE"]["date_column"])
        data = data.drop(columns="meter_id")

        print("The FE columns are: ", data.columns)
        print("The FE columns are: ", data.dtypes)

        return data

    def split_data(data, config):
        # Ensure data is sorted by date
        data = data.sort_values(config["FE"]["date_column"])

        # Calculate the index for the split point
        split_point = int(len(data) * 0.8)  # 80% for training, 20% for testing

        # Assuming the last column is the target, adjust as necessary for your dataset
        X = data.copy()  # Includes target for feature engineering
        y = data.loc[:,config["FE"]["metric"]]   # the last column as the target

        # Split the data into training and testing sets
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        return X_train, X_test, y_train, y_test

    def run_gridsearch(X_train, y_train, FE):
        class CustomFeatureCreator(TransformerMixin, BaseEstimator):
            def __init__(self, config):
                self.config = config
                self.target_column = config["FE"]["metric"]  # Assuming target column name is specified


            def fit(self, X, y=None):
                return self

            def transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    raise TypeError("Input X must be a pandas DataFrame")
                return FE(X, self.config)

        def sklearn_pipe(FE, config, model):
            return Pipeline([
                ('feature_engineering', CustomFeatureCreator(config=config)),
                ('model', model)
            ])
        # Define the model pipeline
        pipeline = Pipeline([
            ('feature_engineering', CustomFeatureCreator(config=config)),
            ('model', LGBMRegressor())
        ])

        # Parameter grid for GridSearch
        param_grid = {
                'model__learning_rate': [0.05, 0.1, 0.2],
                'model__num_leaves': [30, 50, 100, 150],
                'model__max_depth': [5, 10, 15, -1],  # -1 means no limit
                'model__min_child_samples': [10, 20, 30],  # Minimum number of data in one leaf
                'model__min_child_weight': [0.001, 0.01],  # Minimum sum of instance weight (hessian) needed in a child (leaf)
                'model__subsample': [0.8, 0.9, 1.0],  # Subsample ratio of the training instance.
                'model__subsample_freq': [0, 1, 2],  # Frequence of subsample, <=0 means no enable
                'model__colsample_bytree': [0.8, 0.9, 1.0],  # Subsample ratio of columns when constructing each tree.
                'model__reg_alpha': [0.0, 0.1, 0.5],  # L1 regularization term on weights
                'model__reg_lambda': [0.0, 0.1, 0.5],  # L2 regularization term on weights
                'model__boosting_type': ['gbdt', 'dart'],  # 'gbdt', traditional Gradient Boosting Decision Tree. 'dart', Dropouts meet Multiple Additive Regression Trees.
            }

        # Configure GridSearchCV
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train, y_train)
        
         # Extracting best model and predictions
        best_model = grid_search.best_estimator_
        
                
        def align_features(train, test, fill_value=0.0):
            # Get missing columns in the training test
            missing_cols_in_train = set(test.columns) - set(train.columns)
            # Add a missing column in train dataset with default value of 0
            for c in missing_cols_in_train:
                train[c] = fill_value

            # Get missing columns in the test set
            missing_cols_in_test = set(train.columns) - set(test.columns)
            # Add a missing column in test dataset with default value of 0
            for c in missing_cols_in_test:
                test[c] = fill_value

            # Ensure the order of column names in both datasets are the same
            train, test = train.align(test, axis=1)

            return train, test

        # Example usage with your datasets
        #X_train, X_test = align_features(X_train, X_test)
        y_pred = best_model.predict(X_test)  # Predict on unseen test data
        best_model_params = grid_search.best_params_

        # Calculating metrics on the test set
        best_mse = mean_squared_error(y_test, y_pred)
        best_mae = mean_absolute_error(y_test, y_pred)
        best_r2 = r2_score(y_test, y_pred)
        best_rmse = np.sqrt(best_mse)  # Square root of MSE for RMSE

        return best_model_params, best_model, best_mse, best_mae, best_r2, best_rmse, X_train.head(1)
        


    def mlflow_logging(
       best_model_params, best_model, best_mse, best_mae, best_r2, best_rmse,  X):
        """
        Log the best model and relevant metrics to MLflow.

        Args:
            best_model: The best model selected after cross-validation.
            best_params: The best hyperparameters for the model.
            best_mse: Best Mean Squared Error (MSE).
            best_model_type: Type of the best model (e.g., Random Forest, XGBoost).
            X_train, X_test, y_train, y_test: Training and testing data for evaluation.
        """
        print("Logging best model and metrics to MLflow...")
        # Set the tracking URI
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        # Set up the experiment
        experiment_name = "Timeseries forecast-ligtgbm"
        start_time = time.time()

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(best_model_params)

            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mlflow.set_tag("model_registration_date", current_date)
            mlflow.log_metric("mse", best_mse)
            mlflow.log_metric("mae", best_mae)
            mlflow.log_metric("r2_score", best_r2)
            mlflow.log_metric("rmse", best_rmse)

            # Log training time
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_sec", training_time)

            # Log cross-validation performance (mean scores, e.g., for each fold)
            # You can retrieve cross-validation scores from GridSearchCV or RandomizedSearchCV
            if hasattr(best_model, "cv_results_"):
                cv_results = best_model.cv_results_
                for key in cv_results.keys():
                    if "mean_test_score" in key:
                        mlflow.log_metric(f"cv_{key}", cv_results[key].mean())

            # Log model size
            model_size = best_model.__sizeof__()
            mlflow.log_metric("model_size_bytes", model_size)

            # Create an input example and infer the signature
            input_example = X
            signature = infer_signature(X, best_model.predict(X))

            # Log the model with an example input and signature
            if len(X) > 0:
                input_example = X.iloc[0:1]  # Use the first row as an example
                signature = infer_signature(X, best_model.predict(X.iloc[0:1]))
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example
                )
            print("Best model and metrics logged successfully.")

    # def load_yaml_config(file_path):
    #     with open(file_path, "r") as file:
    #         return yaml.safe_load(file)

    # cv_folder = config["info"]["cv_folder"]
    # model_type = config["info"]["model_type"]
    # model_config_folder = config["info"]["model_config_folder"]
    # model_config = load_yaml_config(model_config_path)
    # print(model_config)
    # # Execute pipeline steps
    data = ETL(config)
    # # EDA
    # # perform_eda(data, target_column)
    # date_columns = identify_date_columns(data)
    data = data_prep(data, config)
    config = {
        "FE": {
            "date_column": "DateTime",  # The name of the column containing datetime data
            "metric": "Demand_kW",    # The main metric or column to create features from
            "lag_values": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Lag intervals for lag features
            "window_sizes": [4, 8, 16, 32, 64, 128],    # Window sizes for rolling features
            "rolling_features": ['mean', 'std', 'max', 'min'],  # Types of rolling features to compute
        },
        "model_params": {
            "n_estimators": [50, 100],  # Range of 'n_estimators' for the grid search
            "max_depth": [3, 5],        # Range of 'max_depth' for the grid search
            "learning_rate": [0.01, 0.1]  # Range of 'learning_rate' for the grid search
        },
        "pipeline": {
            "model": "XGBRegressor",  # The type of model to use
            "objective": "reg:squarederror"  # Objective function for the model
        }
    }
    
    X_train, X_test, y_train, y_test = split_data(data, config)
    best_model_params, best_model, best_mse, best_mae, best_r2, best_rmse, X = run_gridsearch(X_train, y_train, FE)
    mlflow_logging(best_model_params, best_model, best_mse, best_mae, best_r2, best_rmse, X)

    # non_numeric_columns = identify_non_numeric_columns_for_model(data)
    # if non_numeric_columns:
    #     X_train, X_test, y_train, y_test = split_data(data, config)
    #     best_model, best_params, best_mse, best_model_type = run_model_with_dynamic_cv(
    #         X_train,
    #         X_test,
    #         y_train,
    #         y_test,
    #         config,
    #         model_config,
    #         cv_folder,
    #         model_type,
    #     )
    #     mlflow_logging(best_model, best_params, best_mse, best_model_type, X_train)
    # else:
    #     print("No non-numeric columns found for model training. Exiting...")


def load_yaml_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # Load pipeline configuration

    config_file_path = "/Users/tkax/dev/aimonetize/WIP/DistroEnergyML/configs/config.yaml"
    model_config_path = "/Users/tkax/dev/aimonetize/WIP/DistroEnergyML/configs/model_config.yaml"
    config = load_config(config_file_path)
    model_type = config["info"]["model_type"]
    model_config_folder = config["info"]["model_config_folder"]
    model_config = load_yaml_config(model_config_path)
    # Run the pipeline
    pipeline(config)
