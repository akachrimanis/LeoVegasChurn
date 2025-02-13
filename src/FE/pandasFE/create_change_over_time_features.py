import pandas as pd
import numpy as np

def create_change_over_time_features(df, player_key_col='player_key', date_col='date', metrics=None, windows=[7, 14, 30, 90]):
    """
    Calculates percentage change in metrics between different time periods.

    Args:
        df: Pandas DataFrame with player_key, date, and metric columns.
        player_key_col: Name of the player key column.
        date_col: Name of the date column.
        metrics: List of metric column names to calculate changes for.
                 If None, defaults to all numeric columns except player_key and date.
        windows: List of time window sizes (in days) for comparisons.

    Returns:
        Pandas DataFrame with added change over time feature columns.
        Returns the original DataFrame if any error occurs.
    """

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([player_key_col, date_col])

        if metrics is None:
            numeric_cols = df.select_dtypes(include='number').columns
            metrics = [col for col in numeric_cols if col not in [player_key_col]]

        for metric in metrics:
            for window in windows:
                # Calculate change over time (percentage change)
                new_col_name_pct_change = f'{metric}_change_{window}d'

                df[new_col_name_pct_change] = df.groupby(player_key_col)[metric].pct_change(periods=window) * 100

                #Handles edge cases (first few rows for each player)
                df[new_col_name_pct_change] = df[new_col_name_pct_change].replace([np.inf, -np.inf], np.nan)


        return df

    except (KeyError, TypeError) as e:
        print(f"Error creating change over time features: {e}")
        return df