import pandas as pd
from feature_engine.transformation import LogTransformer


def apply_log_transformation(data, variables):
    """
    Applies log transformation to the specified variables.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to transform

    Returns:
    - DataFrame with log-transformed variables

    Raises:
    - ValueError: If data is not a DataFrame or variables are not in the DataFrame.
    """
    try:
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        # Apply log transformation
        transformer = LogTransformer(variables=variables)
        transformed_data = transformer.fit_transform(data)

        return transformed_data

    except Exception as e:
        print(f"An error occurred in apply_log_transformation: {e}")
        raise


# feature_engine featuretools featuretools tsfresh
