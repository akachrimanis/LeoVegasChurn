import pandas as pd

def count_id_occurrences(df, id_column, count_column_name='id_count'):
    """
    Adds a new column that counts the number of times each ID appears in the dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.
        id_column (str): The name of the ID column.
        count_column_name (str): The name of the new column with count values (default: 'id_count').

    Returns:
        pd.DataFrame: The DataFrame with the new count column added.
    """
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame.")
    
    # Count occurrences of each ID
    id_counts = df[id_column].value_counts().to_dict()
    
    # Map counts back to the DataFrame
    df[count_column_name] = df[id_column].map(id_counts)
    
    return df
