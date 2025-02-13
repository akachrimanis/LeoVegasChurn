import pandas as pd
from feature_engine.discretisation import ArbitraryDiscretiser


def discretize_with_arbitrary_bins(data, variable, bin_edges):
    """
    Discretizes a variable into bins defined by arbitrary bin edges.

    Parameters:
    - data: DataFrame
    - variable: The column name of the variable to discretize
    - bin_edges: List of bin edges to define the discretization

    Returns:
    - DataFrame with discretized variable
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if variable not in data.columns:
            raise ValueError(f"The variable '{variable}' is not in the DataFrame.")
        if not isinstance(bin_edges, list) or len(bin_edges) < 2:
            raise ValueError("bin_edges must be a list of at least two elements.")

        transformer = ArbitraryDiscretiser(variable=variable, bins=bin_edges)
        discretized_data = transformer.fit_transform(data)

        return discretized_data

    except Exception as e:
        print(f"An error occurred in discretize_with_arbitrary_bins: {e}")
        raise
