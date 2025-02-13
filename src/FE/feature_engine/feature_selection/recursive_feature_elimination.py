from feature_engine.selection import RecursiveFeatureElimination
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def recursive_feature_elimination_selection(
    data, variables, target, estimator=None, n_features_to_select=5
):
    """
    Perform recursive feature elimination using a model.

    Parameters:
    - data: DataFrame
    - variables: List of variable names to evaluate
    - target: Target variable name for evaluation
    - estimator: Model to use for feature importance (default: RandomForestClassifier)
    - n_features_to_select: Number of features to select (default: 5)

    Returns:
    - DataFrame with selected features
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if target not in data.columns:
            raise ValueError(f"The target variable '{target}' is not in the DataFrame.")
        if not all(var in data.columns for var in variables):
            missing_vars = [var for var in variables if var not in data.columns]
            raise ValueError(
                f"The following variables are not in the DataFrame: {missing_vars}"
            )

        if estimator is None:
            estimator = RandomForestClassifier()

        transformer = RecursiveFeatureElimination(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            variables=variables,
        )
        selected_data = transformer.fit_transform(data, data[target])

        return selected_data

    except Exception as e:
        print(f"An error occurred in recursive_feature_elimination_selection: {e}")
        raise
