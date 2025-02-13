from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def select_columns_to_scale(
    X, scaling_criteria="all_numeric", exclude_columns_scaling=[], min_std=0.01
):
    """
    Selects the columns to scale based on the specified criteria.

    Args:
    - X: pandas DataFrame, the feature data
    - scaling_criteria: str, criteria for selecting columns to scale ('all_numeric', 'non_binary', 'high_variance')
    - min_std: float, minimum standard deviation to consider a column for scaling (used in 'high_variance' criteria)

    Returns:
    - columns_to_scale: list of column names that should be scaled
    """
    # Select only numeric columns by default
    if len(exclude_columns_scaling):
        X = X.drop(exclude_columns_scaling, axis=1)
    if scaling_criteria == "all_numeric":
        columns_to_scale = X.select_dtypes(include=["number"]).columns.tolist()
    elif scaling_criteria == "non_binary":
        # Select numeric columns that are not binary (0 or 1)
        columns_to_scale = X.select_dtypes(include=["number"]).columns
        columns_to_scale = [
            col for col in columns_to_scale if not X[col].isin([0, 1]).all()
        ]
    elif scaling_criteria == "high_variance":
        # Select columns with high variance (above a certain threshold)
        columns_to_scale = [
            col
            for col in X.select_dtypes(include=["number"]).columns
            if X[col].std() > min_std
        ]
    else:
        raise ValueError(
            "Invalid scaling_criteria. Choose from 'all_numeric', 'non_binary', or 'high_variance'."
        )

    return columns_to_scale


def scale_features(
    X,
    scaler_type="standard",
    scaling_criteria="all_numeric",
    exclude_columns_scaling=[],
    min_std=0.01,
):
    """
    Scales the features of the DataFrame X using the specified scaler and scaling criteria.

    Args:
    - X: pandas DataFrame, the feature data to scale
    - scaler_type: str, type of scaler to use ('standard' for StandardScaler, 'minmax' for MinMaxScaler)
    - scaling_criteria: str, criteria for selecting columns to scale ('all_numeric', 'non_binary', 'high_variance')
    - min_std: float, minimum standard deviation for selecting columns based on variance (only relevant for 'high_variance')

    Returns:
    - X_scaled: pandas DataFrame, scaled feature data
    """
    try:
        # Select columns to scale based on the given criteria
        columns_to_scale = select_columns_to_scale(
            X, scaling_criteria, exclude_columns_scaling, min_std
        )

        if not columns_to_scale:
            raise ValueError(
                "No columns selected for scaling based on the provided criteria."
            )

        # Apply the selected scaler
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(
                "Invalid scaler_type. Please choose 'standard' or 'minmax'."
            )

        # Scale the selected columns
        X_scaled = X.copy()
        X_scaled[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

        return X_scaled

    except TypeError as e:
        print(f"Error: {e}")
        raise

    except ValueError as e:
        print(f"Error: {e}")
        raise

    except Exception as e:
        print(f"An unexpected error occurred during scaling: {e}")
        raise
