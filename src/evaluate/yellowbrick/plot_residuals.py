from yellowbrick.regressor import ResidualsPlot


def plot_residuals(model, X_train, y_train):
    """
    Plots the residuals of a pre-fitted regression model.
    Skips model fitting if the model is already trained.

    Parameters:
    - model: A fitted regression model.
    - X_train: Features used for training.
    - y_train: True target variable for training.

    Returns:
    - None (plots the residuals).
    """
    try:
        # Create the visualizer and score directly
        visualizer = ResidualsPlot(model)
        visualizer.score(X_train, y_train)
        visualizer.show()
    except Exception as e:
        print(f"An error occurred while plotting the residuals: {e}")
