from feature_engine.selection import DropCorrelatedFeatures
import pandas as pd


def drop_highly_correlated_features(data, variables, threshold=0.9):
    """
    Drop features that are highly correlated with each other.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to check for correlation
    - threshold: Correlation threshold for dropping features (default: 0.9)

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

        transformer = DropCorrelatedFeatures(
            correlation_threshold=threshold, variables=variables
        )
        reduced_data = transformer.fit_transform(data)

        return reduced_data

    except Exception as e:
        print(f"An error occurred in drop_highly_correlated_features: {e}")
        raise
