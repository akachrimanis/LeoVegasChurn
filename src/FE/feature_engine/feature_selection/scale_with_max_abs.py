from feature_engine.scaling import MaxAbsScaler
import pandas as pd


def scale_with_max_abs(data, variables):
    """
    Scales variables by their maximum absolute value.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to scale

    Returns:
    - DataFrame with scaled variables
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        transformer = MaxAbsScaler(variables=variables)
        scaled_data = transformer.fit_transform(data)

        return scaled_data

    except Exception as e:
        print(f"An error occurred in scale_with_max_abs: {e}")
        raise
