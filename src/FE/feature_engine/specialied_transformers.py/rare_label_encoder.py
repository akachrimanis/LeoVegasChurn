from feature_engine.encoding import RareLabelEncoder
import pandas as pd


def encode_rare_labels(data, variable, threshold=0.05):
    """
    Group rare categories into a single label.

    Parameters:
    - data: DataFrame
    - variable: Name of the categorical variable to encode
    - threshold: Frequency threshold for rare categories (default: 0.05)

    Returns:
    - DataFrame with encoded rare labels
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if variable not in data.columns:
            raise ValueError(f"The variable '{variable}' is not in the DataFrame.")

        transformer = RareLabelEncoder(tol=threshold, variables=[variable])
        encoded_data = transformer.fit_transform(data)

        return encoded_data

    except Exception as e:
        print(f"An error occurred in encode_rare_labels: {e}")
        raise
