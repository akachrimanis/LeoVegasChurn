import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
)


def compute_regression_metrics(y_true, y_pred, n_features=None, tau=0.5, delta=1.0):
    """
    Computes various regression metrics.

    Parameters:
    y_true: array-like, true values
    y_pred: array-like, predicted values
    n_features: int, optional, number of features in the model (for Adjusted R²)
    tau: float, optional, quantile for Quantile Loss (default 0.5)
    delta: float, optional, threshold for Huber Loss (default 1.0)

    Returns:
    dict: A dictionary with all the computed metrics
    """

    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)

    # Root Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # R-squared (R²)
    r2 = r2_score(y_true, y_pred)

    # Adjusted R-squared
    adjusted_r2 = None
    if n_features:
        adjusted_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n_features - 1)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Explained Variance Score
    explained_variance = explained_variance_score(y_true, y_pred)

    # Huber Loss
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic_part = np.minimum(abs_error, delta)
    linear_part = abs_error - quadratic_part
    huber_loss = np.mean(0.5 * quadratic_part**2 + delta * linear_part)

    # Quantile Loss
    quantile_loss = np.mean(np.maximum(tau * error, (tau - 1) * error))

    # Log-Cosh Loss
    log_cosh_loss = np.mean(np.log(np.cosh(error)))

    # Median Absolute Error
    median_absolute_error_value = median_absolute_error(y_true, y_pred)

    # Return all metrics in a dictionary
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Adjusted R²": adjusted_r2,
        "MAPE": mape,
        "Explained Variance": explained_variance,
        "Huber Loss": huber_loss,
        "Quantile Loss": quantile_loss,
        "Log-Cosh Loss": log_cosh_loss,
        "Median Absolute Error": median_absolute_error_value,
    }
