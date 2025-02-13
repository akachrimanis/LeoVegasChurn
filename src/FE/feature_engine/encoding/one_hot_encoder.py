import pandas as pd
from feature_engine.encoding import OneHotEncoder


def apply_one_hot_encoding(data, variables):
    """
    Applies one-hot encoding to the specified variables.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to encode

    Returns:
    - DataFrame with one-hot-encoded variables

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

        # Apply one-hot encoding
        encoder = OneHotEncoder(variables=variables)
        encoded_data = encoder.fit_transform(data)

        return encoded_data

    except Exception as e:
        print(f"An error occurred in apply_one_hot_encoding: {e}")
        raise
