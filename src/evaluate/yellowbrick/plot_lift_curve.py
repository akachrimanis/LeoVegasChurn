from yellowbrick.classifier import LiftCurve


def plot_lift_curve(model, X_test, y_test):
    """
    Plots the lift curve for a fitted classification model.

    Parameters:
    - model: A fitted classification model.
    - X_test: Test features.
    - y_test: True target variable for test data.

    Returns:
    - None (plots the lift curve).
    """
    try:
        visualizer = LiftCurve(model)
        visualizer.score(X_test, y_test)
        visualizer.show()
    except Exception as e:
        print(f"An error occurred while plotting the lift curve: {e}")
