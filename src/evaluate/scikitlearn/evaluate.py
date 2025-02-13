from .performance_plots import (
    plot_learning_curve,
    plot_residuals,
    plot_predicted_vs_actual,
    plot_feature_importance,
    plot_shap_values,
    plot_partial_dependence_plot,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_calibration_curve,
)
import mlflow
import os
import time


def performance_plots(
    config, best_model, X_train, y_train, X_test, y_test, predictions
):
    output_path = config["plots"]["output_path"]
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = config["mlflow"]["experiment_name"]  # Example: read from config
    # Check if the experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    output_dir = os.path.join(output_path, str(experiment_id))

    with mlflow.start_run(experiment_id=experiment_id):

        if config["info"]["model_type"] == "regression":
            print("Regression model detected. Plotting regression performance plots...")

            # Plot and log regression plots
            plot_learning_curve(best_model, X_train, y_train, output_dir)
            mlflow.log_artifact(f"{output_dir}/learning_curve.png")

            plot_residuals(y_test, predictions, output_dir)
            mlflow.log_artifact(f"{output_dir}/residuals_plot.png")

            plot_predicted_vs_actual(y_test, predictions, output_dir)
            mlflow.log_artifact(f"{output_dir}/predicted_vs_actual.png")

            plot_feature_importance(best_model, X_train, output_dir)
            # mlflow.log_artifact(f"{output_dir}/feature_importance.png")

            plot_shap_values(best_model, X_train, X_test, output_dir)
            # mlflow.log_artifact(f"{output_dir}/shap_values.png")

            plot_partial_dependence_plot(best_model, X_train, output_dir)
            mlflow.log_artifact(f"{output_dir}/partial_dependence_plot.png")

        elif config["info"]["model_type"] == "classification":
            print(
                "Classification model detected. Plotting Classification performance plots..."
            )

            # Plot and log classification plots
            plot_learning_curve(best_model, X_train, y_train, output_dir)
            mlflow.log_artifact(f"{output_dir}/learning_curve.png")

            plot_confusion_matrix(y_test, predictions, output_dir)
            mlflow.log_artifact(f"{output_dir}/confusion_matrix.png")

            plot_roc_curve(y_test, best_model, X_test, output_dir)
            mlflow.log_artifact(f"{output_dir}/roc_curve.png")

            plot_precision_recall_curve(y_test, best_model, X_test, output_dir)
            mlflow.log_artifact(f"{output_dir}/precision_recall_curve.png")

            plot_calibration_curve(y_test, best_model, X_test, output_dir)
            mlflow.log_artifact(f"{output_dir}/calibration_curve.png")

            plot_shap_values(best_model, X_train, X_test, output_dir)
            mlflow.log_artifact(f"{output_dir}/shap_values.png")

        else:
            print("Model type not recognized. Exiting...")
