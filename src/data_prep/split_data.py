from sklearn.model_selection import train_test_split
import os

print("Current working Dir: ", os.getcwd())

from src.config.load_config import load_config


def split_data(data, config, model_config):
    """
    Split the data into training and testing sets.

    Args:
        - data (pandas DataFrame): The prepared data.
        - config (dict): The configuration dictionary.

    Returns:
        - X_train, X_test, y_train, y_test: The split data for training and testing.
    """
    print("Splitting data into train and test sets...")

    # Extract features and target column
    train_config = config["train"]
    data = data.drop(columns=config["variables"]["drop_cols"])
    y_column = config["variables"]["y"]

    X = data.drop(columns=[y_column])
    y = data[y_column]
    feature_names = X.columns  # List of feature names

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_config["test_size"],
        random_state=train_config["random_state"],
    )
    return X_train, X_test, y_train, y_test, feature_names
