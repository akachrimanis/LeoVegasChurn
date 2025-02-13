from sklearn.impute import SimpleImputer
import pandas as pd


def impute_selected_columns(df, strategy="mean", categorical_placeholder="Unknown"):
    """
    Impute missing values in selected columns based on the column type and chosen strategy.
    For categorical columns, missing values are imputed with an empty string or a placeholder.

    Args:
    - df: pandas DataFrame, the data with missing values to impute
    - strategy: string, the imputation strategy for numerical columns. Options include 'mean', 'median', 'most_frequent', 'constant'
                Default is 'mean'.
    - categorical_placeholder: string, placeholder value for missing categorical data (default is 'Unknown')

    Returns:
    - df_imputed: pandas DataFrame, the DataFrame after imputing missing values
    """
    try:
        # Validate strategy input
        valid_strategies = ["mean", "median", "most_frequent", "constant"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Valid options are: {valid_strategies}")

        # Identify numerical and categorical columns
        numerical_columns = df.select_dtypes(include=["number"]).columns
        categorical_columns = df.select_dtypes(include=["object"]).columns

        # Impute numerical columns (e.g., mean, median)
        if len(numerical_columns) > 0:
            imputer_num = SimpleImputer(strategy=strategy)
            df[numerical_columns] = imputer_num.fit_transform(df[numerical_columns])

        # Impute categorical columns with a placeholder for later dummy encoding
        if len(categorical_columns) > 0:
            # Create a custom imputer for categorical data, replacing missing values with a placeholder
            imputer_cat = SimpleImputer(
                strategy="constant", fill_value=categorical_placeholder
            )
            df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])

        print(f"Missing values imputed successfully using strategy: {strategy}.")
        return df

    except Exception as e:
        print(f"An error occurred while imputing missing values: {e}")
        return df


# Example usage:
# df = pd.read_csv("your_data.csv")  # Load your data
# df_imputed = impute_selected_columns(df, strategy='median')  # Apply median imputation to numerical and placeholder to categorical
