import pandas as pd


def classify_columns(df):
    """
    Classifies columns in a DataFrame into categories based on their data types:
    - Date columns (even those in string format)
    - Numeric columns (int, float)
    - Binary columns (with 2 unique values)
    - Ordinal columns (categorical with an order)

    Parameters:
    - df: The DataFrame to classify

    Returns:
    - A dictionary with the lists of column names for each type of column
    """
    # Initialize lists for each column type
    date_columns = []
    numeric_columns = []
    binary_columns = []
    ordinal_columns = []
    categorical_columns = []

    # Iterate over each column in the DataFrame
    for col in df.columns:
        dtype = df[col].dtype

        # Check for date columns, even if they are in string format
        if pd.api.types.is_string_dtype(dtype):
            try:
                # Try converting the column to datetime
                pd.to_datetime(
                    df[col], errors="raise"
                )  # Will raise an error if conversion fails
                date_columns.append(col)
            except (ValueError, TypeError):
                # If conversion fails, it's not a date column
                pass

        # Check for numeric columns (integers or floats)
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_columns.append(col)

        # Check for binary columns (columns with only 2 unique values)
        elif df[col].nunique() == 2:
            binary_columns.append(col)

        # Check for categorical columns (non-numeric, with an order if applicable)
        elif pd.api.types.is_categorical_dtype(dtype) or df[col].dtype == "object":
            categorical_columns.append(col)

    # Ordinal columns can be defined based on your understanding of the dataset
    # For example, you could manually list them or apply certain heuristics
    # Here we assume categorical columns with more than 2 unique values could be considered ordinal
    for col in categorical_columns:
        if df[col].nunique() > 2:  # Arbitrary threshold for ordinality
            ordinal_columns.append(col)

    # Return a dictionary of the classified columns
    return {
        "date_columns": date_columns,
        "numeric_columns": numeric_columns,
        "binary_columns": binary_columns,
        "ordinal_columns": ordinal_columns,
        "categorical_columns": categorical_columns,
    }
