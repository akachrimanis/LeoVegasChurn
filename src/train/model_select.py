from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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
