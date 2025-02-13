import pandas as pd
from feature_engine.imputation import ArbitraryNumberImputer


def replace_missing_with_arbitrary_number(data, variables, arbitrary_number):
    """
    Replaces missing values with a specified arbitrary number.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to impute
    - arbitrary_number: Number to replace missing values with

    Returns:
    - DataFrame with missing values replaced by the arbitrary number
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = ArbitraryNumberImputer(
            variables=variables, arbitrary_number=arbitrary_number
        )
        imputed_data = transformer.fit_transform(data)

        return imputed_data

    except Exception as e:
        print(f"An error occurred in replace_missing_with_arbitrary_number: {e}")
        raise
