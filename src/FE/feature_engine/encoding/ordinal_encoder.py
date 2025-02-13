from feature_engine.encoding import OrdinalEncoder
import pandas as pd


def encode_with_ordinal(data, variables, encoding_dict=None):
    """
    Encodes categorical variables with ordinal numbers.

    Parameters:
    - data: DataFrame
    - variables: List of categorical variable names to encode
    - encoding_dict: Optional dictionary for custom mapping. Format: {"variable": {"category": value}}

    Returns:
    - DataFrame with encoded variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = OrdinalEncoder(
            encoding_method="ordered", variables=variables, encoding_dict=encoding_dict
        )
        encoded_data = transformer.fit_transform(data)

        return encoded_data

    except Exception as e:
        print(f"An error occurred in encode_with_ordinal: {e}")
        raise
