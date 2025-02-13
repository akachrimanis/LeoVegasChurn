from feature_engine.selection import DropConstantFeatures
import pandas as pd


def drop_constant_value_features(data, variables):
    """
    Drop features that have constant values across all rows.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to check for constant values

    Returns:
    - DataFrame with dropped features
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = DropConstantFeatures(variables=variables)
        reduced_data = transformer.fit_transform(data)

        return reduced_data

    except Exception as e:
        print(f"An error occurred in drop_constant_value_features: {e}")
        raise
