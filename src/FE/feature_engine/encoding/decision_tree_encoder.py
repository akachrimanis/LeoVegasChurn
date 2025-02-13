from feature_engine.encoding import DecisionTreeEncoder
import pandas as pd


def encode_with_decision_tree(data, variables, target, max_depth=3, random_state=None):
    """
    Encodes categorical variables using predictions of a decision tree.

    Parameters:
    - data: DataFrame
    - variables: List of categorical variable names to encode
    - target: Target variable name
    - max_depth: Maximum depth of the decision tree (default: 3)
    - random_state: Random seed for reproducibility (default: None)

    Returns:
    - DataFrame with encoded variables
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

        transformer = DecisionTreeEncoder(
            variables=variables, max_depth=max_depth, random_state=random_state
        )
        encoded_data = transformer.fit_transform(data, data[target])

        return encoded_data

    except Exception as e:
        print(f"An error occurred in encode_with_decision_tree: {e}")
        raise
