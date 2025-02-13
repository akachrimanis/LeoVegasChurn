import pandas as pd

def add_date_range_columns(df, date_column, id_column):
    """
    Adds four columns to the DataFrame:
    - 'start_date': Minimum date for each ID
    - 'end_date': Maximum date for each ID
    - 'max_date': The overall maximum date (constant for all rows)
    - 'min_date': The overall minimum date (constant for all rows)

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_column (str): The name of the date column.
        id_column (str): The name of the ID column.

    Returns:
        pd.DataFrame: The updated DataFrame with four new columns.
    """
    if date_column not in df.columns or id_column not in df.columns:
        raise ValueError(f"Columns '{date_column}' or '{id_column}' not found in DataFrame.")
    
    # Convert to datetime if not already
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Compute start and end dates per ID
    df['start_date'] = df.groupby(id_column)[date_column].transform('min')
    df['end_date'] = df.groupby(id_column)[date_column].transform('max')

    # Compute global min/max dates
    min_date = df[date_column].min()
    max_date = df[date_column].max()

    df['min_date'] = min_date
    df['max_date'] = max_date

    return df
