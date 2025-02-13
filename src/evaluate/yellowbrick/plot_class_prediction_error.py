from yellowbrick.classifier import ClassPredictionError


def plot_class_prediction_error(model, X_test, y_test):
    """
    Plots the class prediction error for a fitted classification model.

    Parameters:
    - model: A fitted classification model.
    - X_test: Test features.
    - y_test: True target variable for test data.

    Returns:
    - None (plots the class prediction error).
    """
    try:
        visualizer = ClassPredictionError(model)
        visualizer.score(X_test, y_test)
        visualizer.show()
    except Exception as e:
        print(f"An error occurred while plotting the class prediction error: {e}")
