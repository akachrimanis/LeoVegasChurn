from feature_engine.selection import DropDuplicateFeatures
import pandas as pd


def drop_duplicate_features(data, variables):
    """
    Drop duplicate features (columns with identical values).

    Parameters:
    - data: DataFrame
    - variables: List of variable names to check for duplicates

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

        transformer = DropDuplicateFeatures(variables=variables)
        reduced_data = transformer.fit_transform(data)

        return reduced_data

    except Exception as e:
        print(f"An error occurred in drop_duplicate_features: {e}")
        raise
