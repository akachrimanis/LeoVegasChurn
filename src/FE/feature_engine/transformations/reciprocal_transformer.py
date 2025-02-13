from feature_engine.transformation import ReciprocalTransformer
import pandas as pd


def apply_reciprocal_transformation(data, variables):
    """
    Applies reciprocal transformation (1/x) to the specified variables.

    Parameters:
    - data: DataFrame
    - variables: List of numerical variable names to transform

    Returns:
    - DataFrame with reciprocal-transformed variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = ReciprocalTransformer(variables=variables)
        transformed_data = transformer.fit_transform(data)

        return transformed_data

    except Exception as e:
        print(f"An error occurred in apply_reciprocal_transformation: {e}")
        raise
