import pandas as pd

def convert_columns_to_datetime(df, columns):
    """
    Converts specified columns in a DataFrame from string to datetime format automatically.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to be converted.

    Returns:
        pd.DataFrame: DataFrame with specified columns converted to datetime where applicable.
    """
    for col in columns:
        try:
            # Attempt conversion
            converted_col = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)

            # Check if conversion was successful (less than 50% NaT after conversion)
            if converted_col.notna().sum() / len(df) > 0.5:
                df[col] = converted_col
                print(f"Column '{col}' successfully converted to datetime.")
            else:
                print(f"Column '{col}' could not be reliably converted to datetime. Skipping.")

        except Exception as e:
            print(f"Error converting column '{col}': {e}")

    return df
