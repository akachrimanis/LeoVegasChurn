from feature_engine.creation import TimeSeriesWindowTransformer
import pandas as pd


def create_time_series_lag_features(data, variables, windows=[1, 2, 3]):
    """
    Create time-windowed lag features for time series data.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to create lag features for
    - windows: List of window sizes for lag features (default: [1, 2, 3])

    Returns:
    - DataFrame with lagged features
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = TimeSeriesWindowTransformer(variables=variables, windows=windows)
        lagged_data = transformer.fit_transform(data)

        return lagged_data

    except Exception as e:
        print(f"An error occurred in create_time_series_lag_features: {e}")
        raise
