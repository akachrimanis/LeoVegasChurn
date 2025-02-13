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
from src.evaluate.evaluate import performance_plots

# Prefect Task to load and prepare data
@task
def load_data(config, model_config):
    logger = get_run_logger()
    logger.info("Starting data loading...")
    cv_folder = config["info"]["cv_folder"]
    model_type = config["info"]["model_type"]
    data = ETL_pickle(config, model_config)
    logger.info("Data loading completed.")
    return data, cv_folder, model_type


# Prefect Task to identify date columns
@task
def identify_dates(data):
    logger = get_run_logger()
    logger.info("Identifying date columns...")
    date_columns = identify_date_columns(data)
    logger.info(f"Date columns identified: {date_columns}")
    return date_columns


# Prefect Task for data preparation
@task
def prepare_data(data, config, model_config):
    logger = get_run_logger()
    logger.info("Starting data preparation...")
    prepared_data = data_prep(data, config, model_config)
    logger.info("Data preparation completed.")
    return prepared_data


# Prefect Task for feature engineering
@task
def feature_engineering(data, config, date_columns, model_config):
    logger = get_run_logger()
    logger.info("Starting feature engineering...")
    engineered_data = FE(data, config, date_columns, model_config)
    logger.info("Feature engineering completed.")
    return engineered_data


# Prefect Task to identify non-numeric columns
@task
def identify_non_numeric_columns(data):
    logger = get_run_logger()
    logger.info("Identifying non-numeric columns...")
    non_numeric_columns = identify_non_numeric_columns_for_model(data)
    logger.info(f"Non-numeric columns identified: {non_numeric_columns}")
    return non_numeric_columns


# Prefect Task for splitting the data into training and testing sets
@task
def split_dataset(data, config, model_config):
    logger = get_run_logger()
    logger.info("Splitting the dataset into training and testing sets...")
    X_train, X_test, y_train, y_test, feature_names = split_data(
        data, config, model_config
    )
    logger.info("Dataset splitting completed.")
    return X_train, X_test, y_train, y_test, feature_names


# Prefect Task to train the model with cross-validation
@task
def train_model(
    X_train, X_test, y_train, y_test, config, model_config, cv_folder, model_type
):
    logger = get_run_logger()
    logger.info("Starting model training with cross-validation...")
    best_model, best_params, best_mse, best_model_type = run_model_with_dynamic_cv(
        X_train, X_test, y_train, y_test, config, model_config, cv_folder, model_type
    )
    logger.info(
        f"Model training completed. Best model type: {best_model_type}, Best MSE: {best_mse}"
    )
    return best_model, best_params, best_mse, best_model_type


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
    logger = get_run_logger()
    logger.info("Logging results to MLflow...")
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
    logger.info("Model logged successfully.")


# Prefect Flow to manage the entire pipeline
@flow(name="Sales Forecasting Pipeline")
def pipeline_flow():
    logger = get_run_logger()
    logger.info("Pipeline started.")

    # Load pipeline configuration
    config_file_path = "/Users/tkax/dev/aimonetize/WIP/ProjectTemplates/predictAPI/sales-forecasting/configs/config.yaml"
    config = load_config(config_file_path)
    model_type = config["info"]["model_type"]
    model_config_folder = config["info"]["model_config_folder"]
    model_config = load_config(os.path.join(model_config_folder, f"{model_type}.yaml"))
    logger.info("Configuration loaded successfully.")

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
        X_train, X_test, y_train, y_test, feature_names = split_dataset(
            data, config, model_config
        )
        best_model, best_params, best_mse, best_model_type = run_model_with_dynamic_cv(
            X_train,
            X_test,
            y_train,
            y_test,
            config,
            model_config,
            cv_folder,
            model_type,
            search_method="grid",
            n_splits=5,
            optimization_metric="mse",
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
        predictions = best_model.predict(X_test)
        performance_plots(
            config, best_model, X_train, y_train, X_test, y_test, predictions
        )

    else:
        logger.warning(
            "No non-numeric columns found for model training. Exiting pipeline..."
        )

    logger.info("Pipeline completed.")


# Run the Prefect flow
if __name__ == "__main__":
    pipeline_flow()
