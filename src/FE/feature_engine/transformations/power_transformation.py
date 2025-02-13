from feature_engine.transformation import PowerTransformer
import pandas as pd


def apply_power_transformation(data, variables, method="yeo-johnson"):
    """
    Applies power transformation (e.g., Box-Cox or Yeo-Johnson) to the specified variables.

    Parameters:
    - data: DataFrame
    - variables: List of numerical variable names to transform
    - method: 'box-cox' or 'yeo-johnson' (default: 'yeo-johnson')

    Returns:
    - DataFrame with power-transformed variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = PowerTransformer(variables=variables, method=method)
        transformed_data = transformer.fit_transform(data)

        return transformed_data

    except Exception as e:
        print(f"An error occurred in apply_power_transformation: {e}")
        raise
