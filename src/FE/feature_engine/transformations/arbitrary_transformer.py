from feature_engine.transformation import ArbitraryTransformer
import pandas as pd


def apply_arbitrary_transformation(data, variables, func):
    """
    Applies a custom mathematical transformation to the specified variables.

    Parameters:
    - data: DataFrame
    - variables: List of numerical variable names to transform
    - func: Function to apply (e.g., lambda x: x**2)

    Returns:
    - DataFrame with transformed variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not callable(func):
            raise ValueError("'func' must be a callable function.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = ArbitraryTransformer(variables=variables, func=func)
        transformed_data = transformer.fit_transform(data)

        return transformed_data

    except Exception as e:
        print(f"An error occurred in apply_arbitrary_transformation: {e}")
        raise
