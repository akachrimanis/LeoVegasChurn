from yellowbrick.classifier import ConfusionMatrix


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plots the confusion matrix for a fitted classification model.

    Parameters:
    - model: A fitted classification model.
    - X_test: Test features.
    - y_test: True target variable for test data.

    Returns:
    - None (plots the confusion matrix).
    """
    try:
        visualizer = ConfusionMatrix(model)
        visualizer.score(X_test, y_test)
        visualizer.show()
    except Exception as e:
        print(f"An error occurred while plotting the confusion matrix: {e}")
