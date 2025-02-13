import os
import sys
import pandas as pd
import warnings

# Suppress FutureWarnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the project root to sys.path
project_root = os.path.abspath(
    "/Users/tkax/dev/aimonetize/WIP/ProjectTemplates/predictAPI/sales-forecasting"
)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.load_config import load_config, load_config_files
from src.ETL.ETL_pickle import ETL_pickle
from src.FE.FE import FE
from src.data_prep.data_prep import data_prep
from src.EDA.EDA import perform_eda
from src.EDA.dates import identify_date_columns
from src.data_prep.identify_cols import identify_non_numeric_columns_for_model
from src.CV.CV import run_model_with_dynamic_cv
from src.mlflow_logging.regression import mlflow_logging
from src.data_prep.split_data import split_data
from src.evaluate.scikitlearn import performance_plots


def pipeline(config, model_config):
    """
    Full pipeline for training and selecting the best model.

    Args:
        - config (dict): The configuration dictionary containing paths, model parameters, etc.
    """
    cv_folder = config["info"]["cv_folder"]
    model_type = config["info"]["model_type"]
    # Execute pipeline steps
    data = ETL_pickle(config, model_config, n_rows=None, save_processed_data=True)
    # EDA
    # perform_eda(data, target_column)
    date_columns = identify_date_columns(data)
    data = data_prep(data, config, model_config)
    data = FE(data, config, date_columns, model_config)

    non_numeric_columns = identify_non_numeric_columns_for_model(data)

    print(f"----- Split Data Columns for training: {data.columns}")
    print(f"----- shape: {data.shape}")

    if len(non_numeric_columns) == 0:
        X_train, X_test, y_train, y_test, feature_names = split_data(
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
        )
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
        predictions = best_model.predict(X_test)
        performance_plots(
            config, best_model, X_train, y_train, X_test, y_test, predictions
        )

    else:
        print("No non-numeric columns found for model training. Exiting...")

    def load_config_files(
        path="/Users/tkax/dev/aimonetize/WIP/ProjectTemplates/predictAPI/sales-forecasting/configs/config.yaml",
    ):
        """
        Load configuration files from the specified path.

        Args:
            - path (str): The path to the configuration files.

        Returns:
            - dict: The configuration dictionary.
        """
        # config_file_path = "/Users/tkax/dev/aimonetize/WIP/ProjectTemplates/predictAPI/sales-forecasting/configs/config.yaml"
        config = load_config(path)
        model_type = config["info"]["model_type"]
        model_config_folder = config["info"]["model_config_folder"]
        model_config = load_config(
            os.path.join(model_config_folder, f"{model_type}.yaml")
        )
        return config, model_config


if __name__ == "__main__":
    # Load pipeline configuration

    config, model_config = load_config_files()
    # Run the pipeline
    pipeline(config, model_config)
