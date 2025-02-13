import pandas as pd

def calculate_cumulative_metrics(df, player_key_col='player_key', date_col='date', metrics=None):
    """Calculates cumulative sums of metrics from start_date to each date for each player.

    Args:
        df: Pandas DataFrame with player_key, date, and metric columns.
        player_key_col: Name of the player key column.
        date_col: Name of the date column.
        metrics: List of metric column names to calculate cumulative sums for.
                 If None, defaults to all numeric columns except player_key and date.

    Returns:
        Pandas DataFrame with added cumulative metric columns.
        Returns the original DataFrame if any error occurs.
    """

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([player_key_col, date_col])

        if metrics is None:
            numeric_cols = df.select_dtypes(include='number').columns
            metrics = [col for col in numeric_cols if col not in [player_key_col]]

        df = pd.merge(df, player_dates, on=player_key_col, how='left')  # Merge with original df

        new_cols = {}

        for metric in metrics:
            new_col_name = f'{metric}_cumulative'
            cumulative_sum = []
            for player in df[player_key_col].unique():
                player_df = df[df[player_key_col] == player].copy()  # Create a copy to avoid SettingWithCopyWarning
                player_df[new_col_name] = 0  # Initialize to 0
                for i in range(len(player_df)):
                    player_df.loc[player_df.index[i], new_col_name] = player_df.loc[player_df.index[:i+1], metric].sum() #Calculate cumulative sum up to current date
                cumulative_sum.extend(player_df[new_col_name].values)
            new_cols[new_col_name] = pd.Series(cumulative_sum)

        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

        return df

    except (KeyError, TypeError) as e:
        print(f"Error creating cumulative features: {e}")
        return df