import pandas as pd

def report_missing_data(df):
    """
    Reports the number of rows with any missing values and prints their indices.

    Args:
    df (pd.DataFrame): The DataFrame to check for missing data.

    Returns:
    None: This function prints results directly and does not return a value.
    """
    # Create a mask where True indicates missing values
    missing_mask = df.isna().any(axis=1)
    
    # Count the number of rows with at least one missing value
    missing_rows_count = missing_mask.sum()
    
    # Get the indices of rows with missing values
    missing_indices = df.index[missing_mask].tolist()
    
    # Print the results
    print(f"Total number of rows with at least one missing value: {missing_rows_count}")
    print(f"Indices of rows with missing data: {missing_indices}")



