import pandas as pd
def create_rolling_features(df, player_key_col='player_key', date_col='date', metrics=None, windows=[7, 14, 30, 90]):
    """
    Calculates rolling averages/sums for specified metrics over different time windows.

    Args:
        df: Pandas DataFrame with player_key, date, and metric columns.
        player_key_col: Name of the player key column.
        date_col: Name of the date column.
        metrics: List of metric column names to calculate rolling features for. 
                 If None, defaults to all numeric columns except player_key and date.
        windows: List of time window sizes (in days) for rolling calculations.

    Returns:
        Pandas DataFrame with added rolling feature columns.
        Returns the original DataFrame if any error occurs.
    """

    try:
        df[date_col] = pd.to_datetime(df[date_col])  # Ensure date is datetime
        df = df.sort_values([player_key_col, date_col]) #Sort by player and date

        if metrics is None:
            numeric_cols = df.select_dtypes(include='number').columns
            metrics = [col for col in numeric_cols if col not in [player_key_col]]

        for metric in metrics:
            for window in windows:
                # Rolling mean (average)
                new_col_name_mean = f'{metric}_rolling_{window}d_mean'
                df[new_col_name_mean] = df.groupby(player_key_col)[metric].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)

                # Rolling sum
                new_col_name_sum = f'{metric}_rolling_{window}d_sum'
                df[new_col_name_sum] = df.groupby(player_key_col)[metric].rolling(window=window, min_periods=1).sum().reset_index(level=0, drop=True)
                
        return df

    except (KeyError, TypeError) as e:
        print(f"Error creating rolling features: {e}")
        return df