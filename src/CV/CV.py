import os
import yaml
import importlib.util
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, classification_report


def load_cv_function(cv_folder, model_type):
    """
    Dynamically load the CV function from the corresponding file in the CV folder.

    Args:
        cv_folder (str): Path to the folder containing CV function files.
        model_type (str): Name of the model type (e.g., regression, classification).

    Returns:
        function: The CV function for the specified model type.
    """
    cv_file = os.path.join(cv_folder, f"{model_type}.py")
    if not os.path.exists(cv_file):
        raise FileNotFoundError(
            f"{model_type}_CV file {cv_file} for model type {model_type} not found."
        )

    spec = importlib.util.spec_from_file_location(f"{model_type}", cv_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, f"CV"):
        raise AttributeError(f"CV function not found in file {cv_file}.")

    return module.CV


def run_model_with_dynamic_cv(
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
):
    """
    Run model training and evaluation using dynamically loaded CV functions.

    Args:
        X_train, X_test, y_train, y_test: Training and testing data.
        config_folder (str): Path to the folder containing YAML config files.
        cv_folder (str): Path to the folder containing CV functions.
        model_type (str): The model type to solve (e.g., "anomaly_detection").
        search_method (str): "grid" or "random" search for hyperparameter tuning.

    Returns:
        dict: Information about the best model, parameters, and performance metrics.
    """
    # Load the YAML configuration for the specified model type
    # Load the appropriate CV function dynamically
    cv_function = load_cv_function(cv_folder, model_type)

    print(f"Running CV for model type: {model_type}")
    results = cv_function(
        X_train, X_test, y_train, y_test, config, model_config, search_method, n_splits
    )

    return results
