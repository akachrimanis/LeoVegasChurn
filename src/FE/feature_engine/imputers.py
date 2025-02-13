import pandas as pd
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer


def impute_missing_values_with_mean(data, variables):
    """
    Imputes missing values in numerical variables with their mean.

    Parameters:
    - data: DataFrame
    - variables: List of numerical variable names to impute

    Returns:
    - DataFrame with imputed variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        imputer = MeanMedianImputer(imputation_method="mean", variables=variables)
        imputed_data = imputer.fit_transform(data)

        return imputed_data

    except Exception as e:
        print(f"An error occurred in impute_missing_values_with_mean: {e}")
        raise


def impute_missing_categorical(data, variables, fill_value="Missing"):
    """
    Imputes missing values in categorical variables with a specified value.

    Parameters:
    - data: DataFrame
    - variables: List of categorical variable names to impute
    - fill_value: Value to replace missing data with (default: "Missing")

    Returns:
    - DataFrame with imputed variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        imputer = CategoricalImputer(fill_value=fill_value, variables=variables)
        imputed_data = imputer.fit_transform(data)

        return imputed_data

    except Exception as e:
        print(f"An error occurred in impute_missing_categorical: {e}")
        raise
