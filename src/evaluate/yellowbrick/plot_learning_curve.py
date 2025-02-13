from yellowbrick.model_selection import LearningCurve


def plot_learning_curve(
    model, X_train, y_train, X_test, y_test, cv=5, scoring="accuracy", train_sizes=None
):
    """
    Plots the learning curve for a pre-fitted model.
    Skips model fitting if the model is already trained.

    Parameters:
    - model: A fitted machine learning model.
    - X_train: Training features.
    - y_train: Training target variable.
    - X_test: Test features.
    - y_test: Test target variable.
    - cv: Number of cross-validation folds.
    - scoring: Metric used for evaluating performance.
    - train_sizes: Sizes of training set to plot (optional).

    Returns:
    - None (plots the learning curve).
    """
    try:
        # Create the visualizer, fit on training data, and score on test data
        visualizer = LearningCurve(
            model, cv=cv, scoring=scoring, train_sizes=train_sizes
        )
        visualizer.score(X_test, y_test)
        visualizer.show()
    except Exception as e:
        print(f"An error occurred while plotting the learning curve: {e}")
