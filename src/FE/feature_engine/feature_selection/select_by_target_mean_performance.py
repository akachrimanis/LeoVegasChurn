from feature_engine.selection import SelectByTargetMeanPerformance
import pandas as pd


def select_features_by_target_mean(data, variables, target):
    """
    Select features based on their predictive performance for a target variable.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to evaluate
    - target: Target variable name for evaluation

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

        transformer = SelectByTargetMeanPerformance(variables=variables, target=target)
        selected_data = transformer.fit_transform(data)

        return selected_data

    except Exception as e:
        print(f"An error occurred in select_features_by_target_mean: {e}")
        raise
