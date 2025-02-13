from feature_engine.imputation import RandomSampleImputer
import pandas as pd


def replace_missing_with_random_sample(data, variables, seed=None):
    """
    Replaces missing values with a random sample from the same variable.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to impute
    - seed: Random seed for reproducibility (default: None)

    Returns:
    - DataFrame with missing values replaced by a random sample
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = RandomSampleImputer(variables=variables, random_state=seed)
        imputed_data = transformer.fit_transform(data)

        return imputed_data

    except Exception as e:
        print(f"An error occurred in replace_missing_with_random_sample: {e}")
        raise
