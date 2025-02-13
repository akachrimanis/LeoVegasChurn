import pandas as pd
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, GroupKFold
from sklearn.utils import resample
from typing import Optional, Union


def random_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
):
    """
    Splits the data randomly into training and test sets.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int, optional): Seed for random number generator.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.

    Raises:
        ValueError: If `test_size` is not between 0 and 1.
    """
    try:
        if not 0 < test_size < 1:
            raise ValueError("`test_size` must be between 0 and 1.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error in random_split: {e}")
        raise


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
):
    """
    Splits the data into training and test sets while preserving the proportion of each class in the target variable.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int, optional): Seed for random number generator.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.

    Raises:
        ValueError: If `test_size` is not between 0 and 1 or if `y` is not categorical.
    """
    try:
        if not 0 < test_size < 1:
            raise ValueError("`test_size` must be between 0 and 1.")

        if y.dtypes not in ["category", "object", "int64"]:
            raise ValueError(
                "Stratified split is typically used with categorical target variables."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error in stratified_split: {e}")
        raise


def k_fold_cross_validation(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Splits the data into `k` subsets for cross-validation and iterates over the splits.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        n_splits (int): Number of splits (folds) for cross-validation.

    Returns:
        List of tuples: Each tuple contains (X_train, X_val, y_train, y_val) for each split.

    Raises:
        ValueError: If `n_splits` is less than 2.
    """
    try:
        if n_splits < 2:
            raise ValueError("`n_splits` must be greater than 1.")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = [
            (
                X.iloc[train_index],
                X.iloc[val_index],
                y.iloc[train_index],
                y.iloc[val_index],
            )
            for train_index, val_index in kf.split(X)
        ]
        return splits
    except Exception as e:
        print(f"Error in k_fold_cross_validation: {e}")
        raise


def time_series_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Performs time series cross-validation, splitting the data into training and validation sets
    where the validation set is always after the training set.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        n_splits (int): Number of splits for cross-validation.

    Returns:
        List of tuples: Each tuple contains (X_train, X_val, y_train, y_val) for each split.

    Raises:
        ValueError: If `n_splits` is less than 2.
    """
    try:
        if n_splits < 2:
            raise ValueError("`n_splits` must be greater than 1.")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = [
            (
                X.iloc[train_index],
                X.iloc[val_index],
                y.iloc[train_index],
                y.iloc[val_index],
            )
            for train_index, val_index in tscv.split(X)
        ]
        return splits
    except Exception as e:
        print(f"Error in time_series_split: {e}")
        raise


def bootstrap_sampling(X: pd.DataFrame, y: pd.Series, n_iterations: int = 1000):
    """
    Generates bootstrap samples by randomly resampling the data with replacement.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        n_iterations (int): Number of bootstrap samples to generate.

    Returns:
        List of tuples: Each tuple contains a bootstrap sample (X_resampled, y_resampled).
    """
    try:
        bootstrap_samples = []
        for _ in range(n_iterations):
            X_resampled, y_resampled = resample(X, y, random_state=42)
            bootstrap_samples.append((X_resampled, y_resampled))
        return bootstrap_samples
    except Exception as e:
        print(f"Error in bootstrap_sampling: {e}")
        raise


def group_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    group_column: str,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
):
    """
    Splits the data based on a grouping column, ensuring that all data from each group is either
    in the training or testing set but not both.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        group_column (str): Name of the column that defines groups (e.g., user ID, region).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int, optional): Seed for random number generator.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.

    Raises:
        KeyError: If `group_column` is not in the DataFrame.
    """
    try:
        if group_column not in X.columns:
            raise KeyError(f"Column `{group_column}` not found in the DataFrame.")

        groups = X[group_column].unique()
        train_groups, test_groups = train_test_split(
            groups, test_size=test_size, random_state=random_state
        )
        train = X[X[group_column].isin(train_groups)]
        test = X[X[group_column].isin(test_groups)]
        y_train = y[X[group_column].isin(train_groups)]
        y_test = y[X[group_column].isin(test_groups)]
        return train, test, y_train, y_test
    except Exception as e:
        print(f"Error in group_based_split: {e}")
        raise
