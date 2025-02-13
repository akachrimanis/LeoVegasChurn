from feature_engine.discretization import EqualFrequencyDiscretiser
import pandas as pd


def equal_frequency_discretise(data, variable, q=4):
    """
    Group numerical variables into bins with equal frequencies.

    Parameters:
    - data: DataFrame
    - variable: Name of the variable to discretize
    - q: Number of quantiles (default is 4, meaning 4 equal bins)

    Returns:
    - DataFrame with discretized variable
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if variable not in data.columns:
            raise ValueError(f"The variable '{variable}' is not in the DataFrame.")

        transformer = EqualFrequencyDiscretiser(q=q, variables=[variable])
        discretized_data = transformer.fit_transform(data)

        return discretized_data

    except Exception as e:
        print(f"An error occurred in equal_frequency_discretise: {e}")
        raise
