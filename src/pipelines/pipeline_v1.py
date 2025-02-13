import os
import sys
import pandas as pd
import numpy as np
import yaml
import joblib
import mlflow
import time
import warnings
import datetime
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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

# Suppress FutureWarnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the project root to sys.path
project_root = os.path.abspath(
    "/Users/tkax/dev/aimonetize/WIP/ProjectTemplates/predictAPI/sales-forecasting"
)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.load_config import load_config
from src.FE.normalize import scale_features, select_columns_to_scale
from src.FE.encoding import prepare_column_lists, encode_columns
from src.FE.date_features import create_extended_date_features
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

        # Drop duplicate rows
        data = data.drop_duplicates()

        # Fill missing values with mean (for simplicity in this example)
        # data.fillna(data.mean(), inplace=True)
        data = convert_integers_to_float(data)
        data = impute_selected_columns(data, strategy="mean")
        # categorical_columns, target_column, categories = prepare_column_lists(df, target_column='target', ordinal_columns=ordinal_columns)
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

        # One-hot encode categorical variables
        scaler_type = config["FE"]["encoding_params"]["scaler_type"]
        scaling_criteria = config["FE"]["encoding_params"]["scaling_criteria"]
        target_column = config["variables"]["y"]
        ordinal_columns = config["data_prep"]["ordinal_columns"]

        # Convert integers to floats where necessary
        data = convert_integers_to_float(data)
        data = create_extended_date_features(
            data, config["data_prep"]["date_columns"][0]
        )
        categorical_columns, target_column, categories = prepare_column_lists(
            data, target_column=target_column, ordinal_columns=ordinal_columns
        )
        data = encode_columns(
            data, categorical_columns, target_column, categories, date_columns
        )

        # Scale features (standard scaling by default)
        try:
            data = scale_features(data, scaler_type, scaling_criteria)
            print("Scaled Data (Standard):\n", data)
        except Exception as e:
            print(f"Failed to scale with StandardScaler: {e}")

        # Save the engineered data
        output_path = config["data_prep"]["engineered_data_path"]
        joblib.dump(data, output_path)
        print(f"Data prepared and saved to {output_path}.")

        return data

    def split_data(data, config):
        """
        Split the data into training and testing sets.

        Args:
            - data (pandas DataFrame): The prepared data.
            - config (dict): The configuration dictionary.

        Returns:
            - X_train, X_test, y_train, y_test: The split data for training and testing.
        """
        print("Splitting data into train and test sets...")

        # Extract features and target column
        train_config = config["train"]
        data = data.drop(columns=config["variables"]["drop_cols"])
        y_column = config["variables"]["y"]

        X = data.drop(columns=[y_column])
        y = data[y_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=train_config["test_size"],
            random_state=train_config["random_state"],
        )

        return X_train, X_test, y_train, y_test

    def CV(X_train, X_test, y_train, y_test, config, search_method="grid"):
        """
        Perform cross-validation and model selection with hyperparameter tuning.

        Args:
            - X_train, X_test, y_train, y_test: The training and testing data.
            - config (dict): The configuration dictionary containing model parameters.
            - search_method (str): Choose between "grid" or "random" search methods.

        Returns:
            - best_model: The best model after hyperparameter tuning.
            - best_params: The best hyperparameters found.
            - best_mse: The best mean squared error from the model.
            - best_model_type: The type of the best model used.
        """
        best_model = None
        best_params = None
        best_mse = float("inf")
        best_model_type = None

        print(
            f"Performing cross-validation and model selection with {search_method} search..."
        )

        for model_type, model_config in config["models"][
            config["info"]["model_family"]
        ].items():
            if model_type in config["train"]["model_types"]:
                print(f"Evaluating model: {model_type}")

                # Dynamically load the model class from the specified library and implementation
                library = model_config["library"]
                implementation = model_config["implementation"]
                module = importlib.import_module(library)
                model_class = getattr(module, implementation)

                # Get hyperparameters from config
                param_grid = model_config["hyperparameters"]

                # Choose search method
                if search_method == "grid":
                    search = GridSearchCV(
                        estimator=model_class(),
                        param_grid=param_grid,
                        cv=5,
                        n_jobs=-1,
                        scoring="neg_mean_squared_error",
                    )
                elif search_method == "random":
                    search = RandomizedSearchCV(
                        estimator=model_class(),
                        param_distributions=param_grid,
                        n_iter=100,
                        cv=5,
                        n_jobs=-1,
                        scoring="neg_mean_squared_error",
                        random_state=42,
                    )

                # Train the model using the search method
                print(f"Training {model_type} model...")
                search.fit(X_train, y_train)

                # Predict and evaluate performance
                predictions = search.best_estimator_.predict(X_test)
                mse = mean_squared_error(y_test, predictions)

                print(f"Best Params: {search.best_params_}, MSE: {mse}")

                # Check if the current model is the best so far
                if mse < best_mse:
                    best_mse = mse
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    best_model_type = model_type  # Save the model type

            print(f"Best model with MSE: {best_mse}")
        else:
            print("No models selected for training. Exiting...")

        return best_model, best_params, best_mse, best_model_type

    def mlflow_logging(
        best_model,
        best_params,
        best_mse,
        best_model_type,
        X_train,
        X_test,
        y_train,
        y_test,
    ):
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

        start_time = time.time()

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_param("best_model_type", best_model_type)
            mlflow.log_params(best_params)

            # Log additional metrics
            predictions = best_model.predict(X_test)
            mse = best_mse
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            rmse = mse**0.5
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mlflow.set_tag("model_registration_date", current_date)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("rmse", rmse)

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
            scaler_type = config["FE"]["encoding_params"]["scaler_type"]
            scaling_criteria = config["FE"]["encoding_params"]["scaling_criteria"]

            # Log feature engineering details
            # If you performed any preprocessing, you can log those parameters here
            mlflow.log_param(
                "scaling_method", scaler_type
            )  # Example scaling method used
            mlflow.log_param(
                "scaling_met", scaling_criteria
            )  # Example scaling method used

            # Create an input example and infer the signature
            input_example = X_train.head(1)
            signature = infer_signature(X_train, best_model.predict(X_train))

            # Log the model with input example and signature
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            print("Best model and metrics logged successfully.")

    def load_yaml_config(file_path):
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    cv_folder = config["info"]["cv_folder"]
    model_type = config["info"]["model_type"]
    model_config_folder = config["info"]["model_config_folder"]
    model_config = load_yaml_config(model_config_path)
    print(model_config)
    # Execute pipeline steps
    data = ETL(config)
    data = data.drop(columns=["product_name"], axis=1)
    # EDA
    # perform_eda(data, target_column)
    date_columns = identify_date_columns(data)
    data = data_prep(data, config)
    data = FE(data, config)

    non_numeric_columns = identify_non_numeric_columns_for_model(data)
    if non_numeric_columns:
        X_train, X_test, y_train, y_test = split_data(data, config)
        best_model, best_params, best_mse, best_model_type = run_model_with_dynamic_cv(
            X_train,
            X_test,
            y_train,
            y_test,
            config,
            model_config,
            cv_folder,
            model_type,
        )
        mlflow_logging(best_model, best_params, best_mse, best_model_type, X_train)
    else:
        print("No non-numeric columns found for model training. Exiting...")


def load_yaml_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # Load pipeline configuration

    config_file_path = "/Users/tkax/dev/aimonetize/WIP/ProjectTemplates/predictAPI/sales-forecasting/configs/config.yaml"
    model_config_path = "/Users/tkax/dev/aimonetize/WIP/ProjectTemplates/predictAPI/sales-forecasting/configs/model_config.yaml"
    config = load_config(config_file_path)
    model_type = config["info"]["model_type"]
    model_config_folder = config["info"]["model_config_folder"]
    model_config = load_yaml_config(model_config_path)
    # Run the pipeline
    pipeline(config)
