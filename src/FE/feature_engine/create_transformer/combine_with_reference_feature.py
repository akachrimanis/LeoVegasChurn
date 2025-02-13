from feature_engine.creation import CombineWithReferenceFeature
import pandas as pd


def combine_with_reference_feature(data, variables, reference_feature, operation="sum"):
    """
    Create new features by combining existing features with a reference feature using mathematical operations.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to combine
    - reference_feature: Name of the reference feature to combine with
    - operation: Mathematical operation to apply ('sum', 'product', 'difference', etc.)

    Returns:
    - DataFrame with new combined features
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )
        if reference_feature not in data.columns:
            raise ValueError(
                f"The reference feature '{reference_feature}' is not in the DataFrame."
            )

        transformer = CombineWithReferenceFeature(
            variables=variables, reference=reference_feature, operation=operation
        )
        new_data = transformer.fit_transform(data)

        return new_data

    except Exception as e:
        print(f"An error occurred in combine_with_reference_feature: {e}")
        raise
