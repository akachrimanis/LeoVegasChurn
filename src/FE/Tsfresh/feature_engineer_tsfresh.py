import tsfresh
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFeatureExtractor
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import numpy as np
# 1. Feature Extraction from Time Series
def extract_time_series_features(df, column_id, column_sort):
    """
    Extracts features from time series data using tsfresh's extract_features function.

    Args:
    df (pd.DataFrame): The input DataFrame with time series data.
    column_id (str): The name of the column that identifies the individual time series.
    column_sort (str): The name of the column that indicates the time ordering of observations.

    Returns:
    pd.DataFrame: The extracted features for each time series.
    """
    try:
        # Extract features
        features = extract_features(df, column_id=column_id, column_sort=column_sort)

        # Impute any missing values in the features
        features = impute(features)

        return features

    except Exception as e:
        print(f"Error in extract_time_series_features: {e}")
        return None


# 2. Comprehensive Feature Extraction
def extract_comprehensive_features(df, column_id, column_sort):
    """
    Extracts a comprehensive set of features from time series data using tsfresh's ComprehensiveFeatureExtractor.

    Args:
    df (pd.DataFrame): The input DataFrame with time series data.
    column_id (str): The name of the column that identifies the individual time series.
    column_sort (str): The name of the column that indicates the time ordering of observations.

    Returns:
    pd.DataFrame: The extracted comprehensive features for each time series.
    """
    try:
        # Sort the DataFrame by the time column
        df = df.sort_values(by=column_sort)

        # Initialize the feature extractor
        extractor = ComprehensiveFeatureExtractor()

        # Extract features
        features = extractor.fit_transform(df)

        # Impute any missing values in the features
        features = impute(features)

        return features

    except Exception as e:
        print(f"Error in extract_comprehensive_features: {e}")
        return None


def extract_custom_features(
    df, feature_list=None, column_id="id", column_time="time", column_value="value"
):
    """
    Extract custom features from a time series dataset using tsfresh's ComprehensiveFeatureExtractor.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the time series data. Must include columns for
      `id`, `time`, and `value`.
    - feature_list (list or None): List of feature categories or individual feature names to include.
      If None, all features are extracted.
    - column_id (str): The column name representing the time series ID (e.g., product or customer ID).
    - column_time (str): The column name representing the time column (timestamps).
    - column_value (str): The column name representing the value of the time series (e.g., sales, stock prices).

    Returns:
    - pd.DataFrame: DataFrame with extracted features.
    """
    try:
        # Initialize the feature extractor
        extractor = ComprehensiveFeatureExtractor()

        # Define default feature extraction parameters
        if feature_list:
            # Customizing the extraction to include only the selected features
            fc_parameters = {feature: None for feature in feature_list}
        else:
            fc_parameters = "all"  # Extract all features by default

        # Extract the features based on the provided column names and feature selection
        features = extractor.fit_transform(
            df.rename(
                columns={column_id: "id", column_time: "time", column_value: "value"}
            ),
            default_fc_parameters=fc_parameters,
        )

        return features
    except Exception as e:
        print(f"Error in extract_custom_features: {e}")
        return None


# 3. Rolling Statistics Feature Extraction
def extract_rolling_features(df, column_id, column_sort, window_size):
    """
    Extracts rolling statistics (e.g., mean, std, etc.) as features from time series data.

    Args:
    df (pd.DataFrame): The input DataFrame with time series data.
    column_id (str): The name of the column that identifies the individual time series.
    column_sort (str): The name of the column that indicates the time ordering of observations.
    window_size (int): The window size for calculating rolling statistics.

    Returns:
    pd.DataFrame: The rolling statistics features for each time series.
    """
    try:
        # Compute rolling mean and rolling standard deviation
        rolling_mean = (
            df.groupby(column_id)[column_sort]
            .rolling(window=window_size)
            .mean()
            .reset_index(name=f"rolling_mean_{window_size}")
        )
        rolling_std = (
            df.groupby(column_id)[column_sort]
            .rolling(window=window_size)
            .std()
            .reset_index(name=f"rolling_std_{window_size}")
        )

        # Merge the rolling features back into the original DataFrame
        features = pd.merge(df, rolling_mean, on=[column_id, column_sort], how="left")
        features = pd.merge(
            features, rolling_std, on=[column_id, column_sort], how="left"
        )

        # Impute any missing values
        features = impute(features)

        return features

    except Exception as e:
        print(f"Error in extract_rolling_features: {e}")
        return None


# 4. Lag Features
def extract_lag_features(df, column_id, column_sort, lag):
    """
    Extracts lag features from time series data (previous time points as features).

    Args:
    df (pd.DataFrame): The input DataFrame with time series data.
    column_id (str): The name of the column that identifies the individual time series.
    column_sort (str): The name of the column that indicates the time ordering of observations.
    lag (int): The lag value to create lagged features.

    Returns:
    pd.DataFrame: The lag features for each time series.
    """
    try:
        # Shift the series to create lagged features
        df[f"lag_{lag}"] = df.groupby(column_id)[column_sort].shift(lag)

        # Impute any missing values
        df = impute(df)

        return df

    except Exception as e:
        print(f"Error in extract_lag_features: {e}")
        return None


# 5. Statistical Features
def extract_statistical_features(
    df, column_id, column_sort, statistic_metrics=["mean", "std", "min", "max", "skew"]
):
    """
    Extracts basic statistical features such as mean, std, max, min, and skew from time series data.

    Args:
    df (pd.DataFrame): The input DataFrame with time series data.
    column_id (str): The name of the column that identifies the individual time series.
    column_sort (str): The name of the column that indicates the time ordering of observations.

    Returns:
    pd.DataFrame: The statistical features for each time series.
    """
    try:
        # Compute basic statistics
        statistics = (
            df.groupby(column_id)[column_sort].agg(statistic_metrics).reset_index()
        )

        return statistics

    except Exception as e:
        print(f"Error in extract_statistical_features: {e}")
        return None


# 6. Time-based Features (e.g., day of week, hour of day)
def extract_time_based_features(df, column_sort):
    """
    Extracts time-based features such as day of the week, hour of the day, etc.

    Args:
    df (pd.DataFrame): The input DataFrame with time series data.
    column_sort (str): The name of the column that indicates the time ordering of observations.

    Returns:
    pd.DataFrame: The time-based features for each time series.
    """
    try:
        # Extract time-based features
        df["hour"] = df[column_sort].dt.hour
        df["day_of_week"] = df[column_sort].dt.dayofweek
        df["month"] = df[column_sort].dt.month
        df["year"] = df[column_sort].dt.year

        return df

    except Exception as e:
        print(f"Error in extract_time_based_features: {e}")
        return None
