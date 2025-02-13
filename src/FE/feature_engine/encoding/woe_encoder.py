from feature_engine.encoding import WoEEncoder
import pandas as pd


def encode_with_woe(data, variables, target):
    """
    Encodes categorical variables with Weight of Evidence (WoE).

    Parameters:
    - data: DataFrame
    - variables: List of categorical variable names to encode
    - target: Target variable name (must be binary)

    Returns:
    - DataFrame with encoded variables
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

        transformer = WoEEncoder(variables=variables)
        encoded_data = transformer.fit_transform(data, data[target])

        return encoded_data

    except Exception as e:
        print(f"An error occurred in encode_with_woe: {e}")
        raise
