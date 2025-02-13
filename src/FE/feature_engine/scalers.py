import pandas as pd
from feature_engine.scaling import MinMaxScaler, StandardScaler


def apply_min_max_scaling(data, variables):
    """
    Applies Min-Max Scaling to the specified variables.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to scale

    Returns:
    - DataFrame with Min-Max Scaled variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        scaler = MinMaxScaler(variables=variables)
        scaled_data = scaler.fit_transform(data)

        return scaled_data

    except Exception as e:
        print(f"An error occurred in apply_min_max_scaling: {e}")
        raise


def apply_standard_scaling(data, variables):
    """
    Applies Standard Scaling (Z-score normalization) to the specified variables.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to scale

    Returns:
    - DataFrame with Standard Scaled variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        scaler = StandardScaler(variables=variables)
        scaled_data = scaler.fit_transform(data)

        return scaled_data

    except Exception as e:
        print(f"An error occurred in apply_standard_scaling: {e}")
        raise
