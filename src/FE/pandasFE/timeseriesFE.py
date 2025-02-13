import pandas as pd



def timeseries_components(df, datetime_col):
    # Set datetime as the index (optional but useful for time series)
    df.set_index(datetime_col, inplace=True)

    # Extract components and convert to categorical data types
    df['month'] = df.index.month.astype('category')
    df['day'] = df.index.day.astype('category')
    df['dayofweek'] = df.index.dayofweek.astype('category')  # Monday=0, Sunday=6
    df['hour'] = df.index.hour.astype('category')
    df_encoded = pd.get_dummies(df, columns=['month', 'day', 'dayofweek', 'hour'])

    # Convert binary columns to float
    for col in df_encoded.columns:
        if 'month_' in col or 'day_' in col or 'dayofweek_' in col or 'hour_' in col:
            df_encoded[col] = df_encoded[col].astype(float)

    # Reset index to move datetime_col back to a column
    df_encoded.reset_index(inplace=True)

    return df_encoded


def timeseries_lags(df, column_id, column_sort, metric, lag_values):
    """
    Extracts lag features from time series data.
    
    Args:
    df (pd.DataFrame): The input DataFrame with time series data.
    column_id (str): The name of the column that identifies the individual time series.
    column_sort (str): The name of the column that indicates the time ordering of observations.
    lag_values (list): A list of lag values to compute.
    """
    
    # Create a copy of the DataFrame
    df_lagged = df.copy()
    
    # Create lag features
    for lag in lag_values:
        df_lagged[f"{metric}_lag{lag}"] = df_lagged.groupby(column_id)[metric].shift(lag)
    # Drop rows that contain any NaN values (especially in the new lag columns)
    #df_lagged = df_lagged.dropna(subset=[f"{metric}_lag{lag}" for lag in lag_values])

    df_lagged.fillna(method='bfill', inplace=True)  # Backfill or another appropriate method

    return df_lagged


def create_rolling_features(df, column, window_size, min_periods=1, feature_list=['mean', 'std', 'max', 'min']):
    """
    Create rolling window features for a given DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the time series data.
    column (str): Column name on which to calculate the rolling features.
    window_size (int): Size of the rolling window.
    min_periods (int): Minimum number of observations in window required to have a value (otherwise result is NA).
    feature_list (list): List of statistical features to calculate (options: 'mean', 'std', 'max', 'min', etc.).

    Returns:
    pd.DataFrame: Original DataFrame with new rolling window features added.
    """
    # Ensure the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame.")

    # Create rolling object
    roller = df[column].rolling(window=window_size, min_periods=min_periods)

    # Generate features
    for feature in feature_list:
        if feature in ['mean', 'std', 'max', 'min', 'sum', 'median']:
            df[f'{column}_rolling_{feature}_{window_size}'] = getattr(roller, feature)()
        else:
            raise ValueError(f"Unsupported feature: {feature}")

    return df


def create_multiple_rolling_features(df, column, window_sizes, features=['mean', 'std', 'max', 'min']):
    """
    Create multiple rolling window features for specified window sizes.

    Args:
    df (pd.DataFrame): DataFrame containing the time series data.
    column (str): Column name on which to calculate the rolling features.
    window_sizes (list): List of integers representing different window sizes.
    features (list): List of statistical features to calculate (e.g., 'mean', 'std', 'max', 'min').

    Returns:
    pd.DataFrame: DataFrame with new rolling window features added.
    """
    for window in window_sizes:
        # Create rolling object for each window size
        roller = df[column].rolling(window=window, min_periods=1)  # min_periods set to 1 to ensure we get values even for small windows

        # Generate features for each statistical measure specified
        for feature in features:
            if feature in ['mean', 'std', 'max', 'min', 'sum']:
                df[f'{column}_rolling_{feature}_{window}'] = getattr(roller, feature)()
            else:
                raise ValueError(f"Unsupported feature: {feature}")

    return df
