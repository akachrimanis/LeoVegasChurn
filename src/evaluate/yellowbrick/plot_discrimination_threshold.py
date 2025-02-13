from yellowbrick.classifier import DiscriminationThreshold


def plot_discrimination_threshold(model, X_test, y_test):
    """
    Plots the discrimination threshold curve for a fitted classification model.

    Parameters:
    - model: A fitted classification model.
    - X_test: Test features.
    - y_test: True target variable for test data.

    Returns:
    - None (plots the discrimination threshold curve).
    """
    try:
        visualizer = DiscriminationThreshold(model)
        visualizer.score(X_test, y_test)
        visualizer.show()
    except Exception as e:
        print(
            f"An error occurred while plotting the discrimination threshold curve: {e}"
        )
