import pandas as pd

def convert_int_to_float(df, exclude_columns=None):
    """
    Converts integer columns in a DataFrame to float, excluding specified columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        exclude_columns (list, optional): List of column names to exclude from conversion. 
                                          Default is None, meaning no exclusions.

    Returns:
        pd.DataFrame: DataFrame with integer columns converted to float, excluding specified columns.
    """
    if exclude_columns is None:
        exclude_columns = []

    # Automatically detect integer columns, excluding specified ones
    int_columns = [col for col in df.select_dtypes(include=['int64', 'int32']).columns if col not in exclude_columns]

    for col in int_columns:
        df[col] = df[col].astype(float)
        print(f"Column '{col}' converted to float.")

    return df
