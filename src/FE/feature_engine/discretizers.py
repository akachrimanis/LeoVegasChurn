import pandas as pd
from feature_engine.discretisation import (
    EqualWidthDiscretiser,
    EqualFrequencyDiscretiser,
)


def apply_equal_width_binning(data, variables, bins=5):
    """
    Discretizes numerical variables into equal-width bins.

    Parameters:
    - data: DataFrame
    - variables: List of numerical variable names to bin
    - bins: Number of bins (default: 5)

    Returns:
    - DataFrame with binned variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        discretiser = EqualWidthDiscretiser(bins=bins, variables=variables)
        binned_data = discretiser.fit_transform(data)

        return binned_data

    except Exception as e:
        print(f"An error occurred in apply_equal_width_binning: {e}")
        raise


def apply_equal_frequency_binning(data, variables, bins=5):
    """
    Discretizes numerical variables into equal-frequency bins.

    Parameters:
    - data: DataFrame
    - variables: List of numerical variable names to bin
    - bins: Number of bins (default: 5)

    Returns:
    - DataFrame with binned variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        discretiser = EqualFrequencyDiscretiser(q=bins, variables=variables)
        binned_data = discretiser.fit_transform(data)

        return binned_data

    except Exception as e:
        print(f"An error occurred in apply_equal_frequency_binning: {e}")
        raise
