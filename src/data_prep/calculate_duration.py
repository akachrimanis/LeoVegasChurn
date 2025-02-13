import pandas as pd

def calculate_duration(df, start_date_col='start_date', end_date_col='end_date', duration_col='duration'):
    """
    Calculates the duration (difference in days) between end_date and start_date.

    Args:
        df (pd.DataFrame): The input DataFrame.
        start_date_col (str): The name of the column with the start date.
        end_date_col (str): The name of the column with the end date.
        duration_col (str): The name of the new column to store the duration.

    Returns:
        pd.DataFrame: The updated DataFrame with the 'duration' column added.
    """
    if start_date_col not in df.columns or end_date_col not in df.columns:
        raise ValueError(f"Columns '{start_date_col}' or '{end_date_col}' not found in DataFrame.")

    # Convert to datetime if not already
    df[start_date_col] = pd.to_datetime(df[start_date_col], errors='coerce')
    df[end_date_col] = pd.to_datetime(df[end_date_col], errors='coerce')

    # Calculate duration in days
    df[duration_col] = (df[end_date_col] - df[start_date_col]).dt.days

    return df
