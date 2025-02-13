import pandas as pd


def identify_date_columns(data: pd.DataFrame) -> list:
    """
    Identifies potential date columns in a DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        list: A list of column names that are likely to contain date values.
    """
    date_columns = []
    for column in data.columns:
        try:
            # Try to convert the column to datetime
            pd.to_datetime(data[column], format="%Y-%m-%d", errors="raise")
            date_columns.append(column)
        except (ValueError, TypeError):
            # If conversion fails, it's not a date column
            continue

    return date_columns


# Identify date columns
# potential_date_columns = identify_date_columns(df)
# print("Potential Date Columns:", potential_date_columns)
