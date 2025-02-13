import numpy as np
import pandas as pd
from feature_engine.creation import CyclicFeatures


def encode_cyclic_features(data, variables, period=24):
    """
    Encode cyclical features like time (e.g., day of the year, month) into sine and cosine transformations.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to encode
    - period: The cycle period, e.g., 24 for hours of the day, 365 for days of the year

    Returns:
    - DataFrame with sine and cosine transformed features
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = CyclicFeatures(variables=variables, period=period)
        cyclic_data = transformer.fit_transform(data)

        return cyclic_data

    except Exception as e:
        print(f"An error occurred in encode_cyclic_features: {e}")
        raise
