from feature_engine.discretisation import GeometricWidthDiscretiser
import pandas as pd


def discretize_with_geometric_width(data, variable, n_bins):
    """
    Discretizes a variable into bins with increasing widths following a geometric sequence.

    Parameters:
    - data: DataFrame
    - variable: The column name of the variable to discretize
    - n_bins: The number of bins to create

    Returns:
    - DataFrame with discretized variable
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if variable not in data.columns:
            raise ValueError(f"The variable '{variable}' is not in the DataFrame.")
        if not isinstance(n_bins, int) or n_bins <= 1:
            raise ValueError("n_bins must be an integer greater than 1.")

        transformer = GeometricWidthDiscretiser(variable=variable, n_bins=n_bins)
        discretized_data = transformer.fit_transform(data)

        return discretized_data

    except Exception as e:
        print(f"An error occurred in discretize_with_geometric_width: {e}")
        raise
