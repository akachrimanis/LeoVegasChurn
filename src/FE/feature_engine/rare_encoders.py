import pandas as pd
from feature_engine.encoding import RareLabelEncoder


def encode_rare_labels(data, variables, threshold=0.05):
    """
    Groups rare labels in categorical variables into a single category.

    Parameters:
    - data: DataFrame
    - variables: List of categorical variable names
    - threshold: Minimum frequency to consider a label as non-rare (default: 0.05)

    Returns:
    - DataFrame with rare labels grouped
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        encoder = RareLabelEncoder(tol=threshold, variables=variables)
        encoded_data = encoder.fit_transform(data)

        return encoded_data

    except Exception as e:
        print(f"An error occurred in encode_rare_labels: {e}")
        raise
