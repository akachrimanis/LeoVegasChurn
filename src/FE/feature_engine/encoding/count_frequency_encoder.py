import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder


def encode_with_count_or_frequency(data, variables, encoding_method="count"):
    """
    Encodes categorical variables using count or frequency.

    Parameters:
    - data: DataFrame
    - variables: List of categorical variable names to encode
    - encoding_method: 'count' or 'frequency' (default: 'count')

    Returns:
    - DataFrame with encoded variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if encoding_method not in ["count", "frequency"]:
            raise ValueError("Invalid encoding method. Use 'count' or 'frequency'.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = CountFrequencyEncoder(
            encoding_method=encoding_method, variables=variables
        )
        encoded_data = transformer.fit_transform(data)

        return encoded_data

    except Exception as e:
        print(f"An error occurred in encode_with_count_or_frequency: {e}")
        raise
