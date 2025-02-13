from feature_engine.selection import SelectBySingleFeaturePerformance
import pandas as pd


def select_features_by_performance(
    data, variables, target, performance_metric="roc_auc"
):
    """
    Select features based on univariate performance metrics.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to evaluate
    - target: Target variable name for evaluation
    - performance_metric: Performance metric to evaluate features (default: 'roc_auc')

    Returns:
    - DataFrame with selected features
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if target not in data.columns:
            raise ValueError(f"The target variable '{target}' is not in the DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = SelectBySingleFeaturePerformance(
            variables=variables, target=target, scoring=performance_metric
        )
        selected_data = transformer.fit_transform(data)

        return selected_data

    except Exception as e:
        print(f"An error occurred in select_features_by_performance: {e}")
        raise
