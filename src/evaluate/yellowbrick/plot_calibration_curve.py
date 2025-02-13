from yellowbrick.model_selection import ValidationCurve


def plot_validation_curve(model, param_name, param_range, X_train, y_train):
    """
    Plots the validation curve for a model with a specified parameter.

    Parameters:
    - model: A classification model.
    - param_name: The name of the parameter to tune.
    - param_range: A list of values for the parameter.
    - X_train: Training features.
    - y_train: True target variable for training data.

    Returns:
    - None (plots the validation curve).
    """
    try:
        visualizer = ValidationCurve(
            model, param_name=param_name, param_range=param_range
        )
        visualizer.fit(X_train, y_train)
        visualizer.finalize(fig_size=(8, 6))  # Adjust figure size (optional)
        visualizer.show()
    except Exception as e:
        print(f"An error occurred while plotting the validation curve: {e}")
