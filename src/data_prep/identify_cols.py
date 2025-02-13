import pandas as pd


def identify_non_numeric_columns_for_model(data):
    """
    Identify non-numeric columns in the dataset that can be used for model training.
    This includes categorical columns and date columns.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        list: List of non-numeric columns suitable for model training.
    """
    # Identify non-numeric columns (exclude float and int types)
    non_numeric_columns = data.select_dtypes(exclude=["number"]).columns.tolist()

    # Filter out columns that might be irrelevant, like 'id' or 'name'
    irrelevant_columns = [
        "id",
        "name",
        "timestamp",
        "date",
    ]  # Add other irrelevant column names here
    non_numeric_columns = [
        col for col in non_numeric_columns if col.lower() not in irrelevant_columns
    ]

    print("Non-numeric columns that can be used for model training:")
    print(non_numeric_columns)

    return non_numeric_columns
