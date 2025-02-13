from yellowbrick.classifier import CumulativeGain


def plot_cumulative_gain(model, X_test, y_test):
    """
    Plots the cumulative gains curve for a fitted classification model.

    Parameters:
    - model: A fitted classification model.
    - X_test: Test features.
    - y_test: True target variable for test data.

    Returns:
    - None (plots the cumulative gains curve).
    """
    try:
        visualizer = CumulativeGain(model)
        visualizer.score(X_test, y_test)
        visualizer.show()
    except Exception as e:
        print(f"An error occurred while plotting the cumulative gains curve: {e}")
