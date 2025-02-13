from yellowbrick.classifier import PrecisionRecallCurve


def plot_precision_recall_curve(model, X_test, y_test):
    """
    Plots the precision-recall curve for a fitted classification model.
    Skips model fitting if the model is already trained.

    Parameters:
    - model: A fitted classification model.
    - X_test: Test features.
    - y_test: True target variable for test data.

    Returns:
    - None (plots the precision-recall curve).
    """
    try:
        # Create the visualizer and score directly
        visualizer = PrecisionRecallCurve(model)
        visualizer.score(X_test, y_test)
        visualizer.show()
    except Exception as e:
        print(f"An error occurred while plotting the precision-recall curve: {e}")
