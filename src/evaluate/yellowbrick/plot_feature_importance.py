from yellowbrick.model_selection import FeatureImportances


def plot_feature_importance(model, X_train, y_train):
    """
    Plots the feature importance of a pre-fitted model.
    Skips model fitting if the model is already trained.

    Parameters:
    - model: A fitted machine learning model.
    - X_train: Features used for training.
    - y_train: True target variable for training.

    Returns:
    - None (plots the feature importances).
    """
    try:
        # Create the visualizer and show the importance directly
        visualizer = FeatureImportances(model)
        visualizer.score(X_train, y_train)
        visualizer.show()
    except Exception as e:
        print(f"An error occurred while plotting the feature importance: {e}")
