import pandas as pd
from feature_engine.creation import MathFeatures


def create_math_features(data, variables, operation="sum"):
    """
    Create new features by applying mathematical operations on existing ones.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to apply operations on
    - operation: Mathematical operation to apply ('sum', 'product', 'difference', etc.)

    Returns:
    - DataFrame with new mathematical features
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = MathFeatures(variables=variables, operation=operation)
        new_data = transformer.fit_transform(data)

        return new_data

    except Exception as e:
        print(f"An error occurred in create_math_features: {e}")
        raise
