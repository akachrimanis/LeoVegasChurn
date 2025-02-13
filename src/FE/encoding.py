import pandas as pd
import category_encoders as ce


def one_hot_encode(df, categorical_columns):
    """
    One-hot encodes the categorical columns in the dataframe.

    Args:
    - df: pandas DataFrame, the dataset with categorical columns.
    - categorical_columns: list, the names of columns to be one-hot encoded.

    Returns:
    - df_encoded: pandas DataFrame, the dataset with one-hot encoded columns.
    """
    try:
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
        print("One-hot encoding completed successfully.")
        return df_encoded
    except Exception as e:
        print(f"Error during one-hot encoding: {e}")
        return df


from sklearn.preprocessing import LabelEncoder


def label_encode(df, categorical_columns):
    """
    Label encodes the categorical columns in the dataframe.

    Args:
    - df: pandas DataFrame, the dataset with categorical columns.
    - categorical_columns: list, the names of columns to be label encoded.

    Returns:
    - df_encoded: pandas DataFrame, the dataset with label encoded columns.
    """
    try:
        encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = encoder.fit_transform(df[col].astype(str))
        print("Label encoding completed successfully.")
        return df
    except Exception as e:
        print(f"Error during label encoding: {e}")
        return df


def target_encode(df, categorical_columns, target_column):
    """
    Target encodes the categorical columns by replacing categories with the mean of the target variable.

    Args:
    - df: pandas DataFrame, the dataset with categorical columns.
    - categorical_columns: list, the names of columns to be target encoded.
    - target_column: string, the name of the target column.

    Returns:
    - df_encoded: pandas DataFrame, the dataset with target encoded columns.
    """
    try:
        for col in categorical_columns:
            encoding_map = df.groupby(col)[target_column].mean()
            df[col] = df[col].map(encoding_map)
        print("Target encoding completed successfully.")
        return df
    except Exception as e:
        print(f"Error during target encoding: {e}")
        return df


from sklearn.preprocessing import OrdinalEncoder


def ordinal_encode(df, categorical_columns, categories=None):
    """
    Ordinal encodes the categorical columns in the dataframe.

    Args:
    - df: pandas DataFrame, the dataset with categorical columns.
    - categorical_columns: list, the names of columns to be ordinal encoded.
    - categories: list, the order of categories (optional).

    Returns:
    - df_encoded: pandas DataFrame, the dataset with ordinal encoded columns.
    """
    try:
        encoder = OrdinalEncoder(categories=categories)
        for col in categorical_columns:
            df[col] = encoder.fit_transform(df[[col]])
        print("Ordinal encoding completed successfully.")
        return df
    except Exception as e:
        print(f"Error during ordinal encoding: {e}")
        return df


def binary_encode(df, categorical_columns):
    """
    Binary encodes the categorical columns in the dataframe using the category_encoders library.

    Args:
    - df: pandas DataFrame, the dataset with categorical columns.
    - categorical_columns: list, the names of columns to be binary encoded.

    Returns:
    - df_encoded: pandas DataFrame, the dataset with binary encoded columns.
    """
    try:
        encoder = ce.BinaryEncoder(cols=categorical_columns)
        df_encoded = encoder.fit_transform(df)
        print("Binary encoding completed successfully.")
        return df_encoded
    except Exception as e:
        print(f"Error during binary encoding: {e}")
        return df


def frequency_encode(df, categorical_columns):
    """
    Frequency encodes the categorical columns by replacing categories with their frequency in the dataset.

    Args:
    - df: pandas DataFrame, the dataset with categorical columns.
    - categorical_columns: list, the names of columns to be frequency encoded.

    Returns:
    - df_encoded: pandas DataFrame, the dataset with frequency encoded columns.
    """
    try:
        for col in categorical_columns:
            freq_map = df[col].value_counts() / len(df)
            df[col] = df[col].map(freq_map)
        print("Frequency encoding completed successfully.")
        return df
    except Exception as e:
        print(f"Error during frequency encoding: {e}")
        return df


def encode_columns(
    df,
    categorical_columns=None,
    target_column=None,
    categories=None,
    date_columns=None,
    ordinal_columns=None,
):
    """
    Automatically applies the appropriate encoding method based on column types:
    - One-Hot Encoding for nominal variables
    - Ordinal Encoding for ordinal variables
    - Binary Encoding for binary variables

    Args:
    - df: pandas DataFrame, the dataset containing categorical features
    - categorical_columns: list, the names of columns to be encoded (optional)
    - target_column: string, the target column for target encoding (optional)
    - categories: dict, the order of categories for ordinal encoding (optional)

    Returns:
    - df_encoded: pandas DataFrame, the dataset with encoded columns
    """

    # If categorical_columns is not provided, select all object dtype columns
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    if ordinal_columns is None:
        ordinal_columns = []  # Ensure ordinal_columns is an empty list if it's None

    # Helper function to determine if a column is binary
    def is_binary(df, col):
        try:
            if col in df.columns:
                return len(df[col].dropna().unique()) == 2
        except Exception as e:
            print(f"Error during binary encoding: {e}")
            return False

    df_encoded = df.copy()

    if len(categorical_columns) > 0:
        for col in categorical_columns:
            if col in df.columns:
                if is_binary(df_encoded, col):
                    print(f"Binary encoding {col}")
                    df_encoded = one_hot_encode(df_encoded, [col])
                elif df_encoded[col].nunique():
                    print(f"One-hot encoding {col}")
                    df_encoded = one_hot_encode(df_encoded, [col])
                if col in ordinal_columns:
                    print(f"Ordinal encoding {col}")
                    df_encoded = ordinal_encode(df_encoded, [col], categories)

    binary_columns = df_encoded.select_dtypes(include=["bool"]).columns
    df_encoded[binary_columns] = df_encoded[binary_columns].astype(int)

    return df_encoded


def prepare_column_lists(df, target_column=None, ordinal_columns=None):
    """
    Prepares the lists of columns for encoding: categorical columns, target column, and categories (for ordinal encoding).

    Args:
    - df: pandas DataFrame, the dataset.
    - target_column: string (optional), the name of the target column (used for target encoding).
    - ordinal_columns: dict (optional), the dictionary where keys are column names and values are lists of categories
                        to specify the order for ordinal encoding.

    Returns:
    - categorical_columns: list of strings, the names of categorical columns to be encoded.
    - target_column: string, the name of the target column (if provided).
    - categories: dict, the order of categories for ordinal encoding.
    """
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # If a target column is provided, exclude it from the categorical columns list
    if target_column:
        categorical_columns = [
            col for col in categorical_columns if col != target_column
        ]

    # Prepare categories for ordinal encoding (if needed)
    categories = {}
    if ordinal_columns:
        for col, cat_list in ordinal_columns.items():
            if col in categorical_columns:
                categories[col] = cat_list
    print("Categorical columns:", categorical_columns)
    return categorical_columns, target_column, categories


# categorical_columns, target_column, categories = prepare_column_lists(df, target_column='target', ordinal_columns=ordinal_columns)
