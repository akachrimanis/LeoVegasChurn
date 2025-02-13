import mlflow
import time
import datetime
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    mean_squared_error,
)
import pandas as pd


def mlflow_logging(
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
    """
    Log the best model and relevant metrics to MLflow.

    Args:
        best_model: The best model selected after cross-validation.
        best_params: The best hyperparameters for the model.
        best_mse: Best Mean Squared Error (MSE).
        best_model_type: Type of the best model (e.g., Random Forest, XGBoost).
        X_train, X_test, y_train, y_test: Training and testing data for evaluation.
        config: Configuration dictionary containing MLflow parameters.
    """
    print("Logging best model and metrics to MLflow...")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = config["mlflow"]["experiment_name"]  # Example: read from config

    # Check if the experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    start_time = time.time()

    with mlflow.start_run(experiment_id=experiment_id):
        # Log hyperparameters
        mlflow.log_param("best_model_type", best_model_type)
        mlflow.log_params(best_params)

        # Log additional metrics
        predictions = best_model.predict(X_test)
        mse = best_mse
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = mse**0.5
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_tag("model_registration_date", current_date)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)

        # Log training time
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_sec", training_time)

        # Log cross-validation performance (if available)
        if hasattr(best_model, "cv_results_"):
            cv_results = best_model.cv_results_
            for key in cv_results.keys():
                if "mean_test_score" in key:
                    mlflow.log_metric(f"cv_{key}", cv_results[key].mean())

        # Log model size
        model_size = best_model.__sizeof__()
        mlflow.log_metric("model_size_bytes", model_size)

        # Log feature engineering details
        scaler_type = config["FE"]["encoding_params"]["scaler_type"]
        scaling_criteria = config["FE"]["encoding_params"]["scaling_criteria"]
        mlflow.log_param("scaling_method", scaler_type)
        mlflow.log_param("scaling_met", scaling_criteria)

        # Create an input example and infer the signature
        # Replace X_train with a DataFrame if it's a NumPy array
        if isinstance(X_train, pd.DataFrame):
            input_example = X_train.head(1)
        else:
            # Provide column names if available, e.g., config["column_names"]
            column_names = config.get(
                "column_names", [f"feature_{i}" for i in range(X_train.shape[1])]
            )
            input_example = pd.DataFrame(X_train, columns=column_names).head(1)

        signature = infer_signature(X_train, best_model.predict(X_train))

        # Log the model with input example and signature
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        print("Best model and metrics logged successfully.")
