import os
import sys
import pandas as pd
import warnings
from prefect import flow, task, get_run_logger

# Suppress FutureWarnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the project root to sys.path
project_root = os.path.abspath(
    "/Users/tkax/dev/aimonetize/WIP/ProjectTemplates/predictAPI/sales-forecasting"
)

if project_root not in sys.path:
    sys.path.append(project_root)

# Importing the necessary modules
from src.config.load_config import load_config
from src.ETL.ETL_pickle import ETL_pickle
from src.FE.FE import FE
from src.data_prep.data_prep import data_prep
from src.EDA.EDA import perform_eda
from src.EDA.dates import identify_date_columns
from src.data_prep.identify_cols import identify_non_numeric_columns_for_model
from src.CV.CV import run_model_with_dynamic_cv
from src.mlflow_logging.regression import mlflow_logging
from src.data_prep.split_data import split_data

# Prefect Task to load and prepare data
@task
def load_data(config, model_config):
    cv_folder = config["info"]["cv_folder"]
    model_type = config["info"]["model_type"]
    data = ETL_pickle(config, model_config)
    return data, cv_folder, model_type


# Prefect Task to identify date columns
@task
def identify_dates(data):
    date_columns = identify_date_columns(data)
    return date_columns


# Prefect Task for data preparation
@task
def prepare_data(data, config, model_config):
    return data_prep(data, config, model_config)


# Prefect Task for feature engineering
@task
def feature_engineering(data, config, date_columns, model_config):
    return FE(data, config, date_columns, model_config)


# Prefect Task to identify non-numeric columns
@task
def identify_non_numeric_columns(data):
    return identify_non_numeric_columns_for_model(data)


# Prefect Task for splitting the data into training and testing sets
@task
def split_dataset(data, config, model_config):
    return split_data(data, config, model_config)


# Prefect Task to train the model with cross-validation
@task
def train_model(
    X_train, X_test, y_train, y_test, config, model_config, cv_folder, model_type
):
    return run_model_with_dynamic_cv(
        X_train, X_test, y_train, y_test, config, model_config, cv_folder, model_type
    )


# Prefect Task to log the model and results in MLflow
@task
def log_results(
    best_model,
    best_params,
    best_mse,
    best_model_type,
    X_train,
    X_test,
    y_train,
    y_test,
    config,
):
    mlflow_logging(
        best_model,
        best_params,
        best_mse,
        best_model_type,
        X_train,
        X_test,
        y_train,
        y_test,
        config,
    )
    return "Model logged successfully"


# Prefect Flow to manage the entire pipeline
@flow(name="Sales Forecasting Pipeline")
def pipeline_flow():
    # Load pipeline configuration
    config_file_path = "/Users/tkax/dev/aimonetize/WIP/ProjectTemplates/predictAPI/sales-forecasting/configs/config.yaml"
    config = load_config(config_file_path)
    model_type = config["info"]["model_type"]
    model_config_folder = config["info"]["model_config_folder"]
    model_config = load_config(os.path.join(model_config_folder, f"{model_type}.yaml"))

    # Load and prepare data
    data, cv_folder, model_type = load_data(config, model_config)

    # Identify date columns
    date_columns = identify_dates(data)

    # Prepare the data
    data = prepare_data(data, config, model_config)

    # Perform feature engineering
    data = feature_engineering(data, config, date_columns, model_config)

    # Identify non-numeric columns
    non_numeric_columns = identify_non_numeric_columns(data)

    # If there are no non-numeric columns, split the data, train the model, and log the results
    if len(non_numeric_columns) == 0:
        X_train, X_test, y_train, y_test = split_dataset(data, config, model_config)
        best_model, best_params, best_mse, best_model_type = train_model(
            X_train,
            X_test,
            y_train,
            y_test,
            config,
            model_config,
            cv_folder,
            model_type,
        )
        log_results(
            best_model,
            best_params,
            best_mse,
            best_model_type,
            X_train,
            X_test,
            y_train,
            y_test,
            config,
        )
    else:
        print("No non-numeric columns found for model training. Exiting...")


# Run the Prefect flow
if __name__ == "__main__":
    pipeline_flow()
