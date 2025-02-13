import pandas as pd
from feature_engine.selection import DropFeatures


def drop_features(data, variables):
    """
    Drops the specified variables from the dataset.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to drop

    Returns:
    - DataFrame with specified variables removed

    Raises:
    - ValueError: If data is not a DataFrame or variables are not in the DataFrame.
    """
    try:
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        # Drop specified features
        selector = DropFeatures(features_to_drop=variables)
        reduced_data = selector.fit_transform(data)

        return reduced_data

    except Exception as e:
        print(f"An error occurred in drop_features: {e}")
        raise
