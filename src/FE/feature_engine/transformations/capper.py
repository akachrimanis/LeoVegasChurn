from feature_engine.transformation import Capper
import pandas as pd


def cap_values(data, variables, min_cap=None, max_cap=None):
    """
    Caps variable values at specified minimum and maximum thresholds.

    Parameters:
    - data: DataFrame
    - variables: List of numerical variable names to cap
    - min_cap: Minimum value to cap variables at (default: None)
    - max_cap: Maximum value to cap variables at (default: None)

    Returns:
    - DataFrame with capped variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = Capper(variables=variables, max_cap=max_cap, min_cap=min_cap)
        capped_data = transformer.fit_transform(data)

        return capped_data

    except Exception as e:
        print(f"An error occurred in cap_values: {e}")
        raise
