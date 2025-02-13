import pandas as pd


def convert_integers_to_float(df):
    """
    Identifies all integer columns in the DataFrame and converts them to float64.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with integer columns converted to float64.
    """
    # Identify integer columns and convert them to float64
    df[df.select_dtypes(include=["int"]).columns] = df.select_dtypes(
        include=["int"]
    ).astype("float64")

    return df
