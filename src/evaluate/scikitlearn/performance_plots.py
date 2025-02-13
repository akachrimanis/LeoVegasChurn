import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay
import shap
import pandas as pd
import numpy as np

# 1. Learning Curves
def plot_learning_curve(best_model, X_train, y_train, output_dir):
    """
    Plots the learning curve for a model to visualize training and validation score
    as the training size increases. Helps diagnose if the model is underfitting or overfitting.

    Args:
    best_model: Trained machine learning model.
    X_train: Feature data for training.
    y_train: Target labels for training.
    output_dir: Directory to save the plot.

    Returns:
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_sizes, train_scores, test_scores = learning_curve(
            best_model,
            X_train,
            y_train,
            cv=5,
            n_jobs=1,
            train_sizes=np.linspace(0.1, 1.0, 10),
        )

        plt.figure()
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
        plt.plot(train_sizes, test_scores.mean(axis=1), label="Cross-validation score")
        plt.xlabel("Training size")
        plt.ylabel("Score")
        plt.title("Learning Curves")
        plt.legend()

        plot_file_path = os.path.join(output_dir, "learning_curve.png")
        plt.savefig(plot_file_path)
        plt.close()
    except Exception as e:
        print(f"Error in plot_learning_curve: {e}")


# 2. Confusion Matrix
def plot_confusion_matrix(y_test, predictions, output_dir):
    """
    Plots the confusion matrix to evaluate the performance of a classification model.
    Visualizes the true vs predicted labels for binary or multiclass classification.

    Args:
    y_test: True target labels.
    predictions: Predicted labels from the model.
    output_dir: Directory to save the plot.

    Returns:
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")

        plot_file_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(plot_file_path)
        plt.close()
    except Exception as e:
        print(f"Error in plot_confusion_matrix: {e}")


# 3. ROC Curve
def plot_roc_curve(y_test, best_model, X_test, output_dir):
    """
    Plots the ROC curve to evaluate the classification model's performance.
    Shows the trade-off between true positive rate and false positive rate.

    Args:
    y_test: True target labels.
    best_model: Trained classification model.
    X_test: Feature data for testing.
    output_dir: Directory to save the plot.

    Returns:
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        plot_file_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(plot_file_path)
        plt.close()
    except Exception as e:
        print(f"Error in plot_roc_curve: {e}")


# 4. Precision-Recall Curve
def plot_precision_recall_curve(y_test, best_model, X_test, output_dir):
    """
    Plots the precision-recall curve, useful for evaluating performance on imbalanced datasets.

    Args:
    y_test: True target labels.
    best_model: Trained classification model.
    X_test: Feature data for testing.
    output_dir: Directory to save the plot.

    Returns:
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        precision, recall, _ = precision_recall_curve(
            y_test, best_model.predict_proba(X_test)[:, 1]
        )

        plt.figure()
        plt.plot(recall, precision, color="blue", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")

        plot_file_path = os.path.join(output_dir, "precision_recall_curve.png")
        plt.savefig(plot_file_path)
        plt.close()
    except Exception as e:
        print(f"Error in plot_precision_recall_curve: {e}")


# 5. Feature Importance Plot
def plot_feature_importance(best_model, X_train, output_dir):
    """
    Plots the feature importance for tree-based models (e.g., Random Forest, Gradient Boosting).
    Helps identify which features are most influential in the model's predictions.

    Args:
    best_model: Trained model with feature importances.
    X_train: Feature data for training.
    output_dir: Directory to save the plot.

    Returns:
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(best_model, "feature_importances_"):
            feature_importances = pd.Series(
                best_model.feature_importances_, index=X_train.columns
            )
            feature_importances = feature_importances.sort_values(ascending=False)

            plt.figure(figsize=(10, 6))
            feature_importances.plot(kind="bar")
            plt.title("Feature Importances")
            plt.xlabel("Feature")
            plt.ylabel("Importance")

            plot_file_path = os.path.join(output_dir, "feature_importance.png")
            plt.savefig(plot_file_path)
            plt.close()
        else:
            print("Model does not have feature_importances_.")
    except Exception as e:
        print(f"Error in plot_feature_importance: {e}")


# 6. Residuals Plot
def plot_residuals(y_test, predictions, output_dir):
    """
    Plots the residuals to evaluate the model's prediction errors. Residuals should be randomly distributed for good model fit.

    Args:
    y_test: True target labels.
    predictions: Predicted labels from the model.
    output_dir: Directory to save the plot.

    Returns:
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        residuals = y_test - predictions

        plt.figure()
        plt.scatter(predictions, residuals)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predictions")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predictions")

        plot_file_path = os.path.join(output_dir, "residuals_plot.png")
        plt.savefig(plot_file_path)
        plt.close()
    except Exception as e:
        print(f"Error in plot_residuals: {e}")


# 7. Predicted vs Actual Plot
def plot_predicted_vs_actual(y_test, predictions, output_dir):
    """
    Plots the predicted vs actual values to visually assess model performance.

    Args:
    y_test: True target labels.
    predictions: Predicted labels from the model.
    output_dir: Directory to save the plot.

    Returns:
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure()
        plt.scatter(y_test, predictions)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color="red",
            lw=2,
        )
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")

        plot_file_path = os.path.join(output_dir, "predicted_vs_actual.png")
        plt.savefig(plot_file_path)
        plt.close()
    except Exception as e:
        print(f"Error in plot_predicted_vs_actual: {e}")


# 8. SHAP Values
def plot_shap_values(best_model, X_train, X_test, output_dir):
    """
    Plots SHAP values to explain the predictions made by a model.
    SHAP (Shapley Additive Explanations) helps interpret the influence of each feature on predictions.

    Args:
    best_model: Trained model (e.g., tree-based).
    X_train: Feature data for training.
    X_test: Feature data for testing.
    output_dir: Directory to save the plot.

    Returns:
    None
    """
    try:
        model = best_model.named_steps[
            "model"
        ]  # Access the 'model' part of the pipeline

        if hasattr(model, "predict"):
            print("Model is compatible with SHAP.")
        else:
            raise ValueError("The model is not compatible with SHAP explainer.")

        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        plt.figure()
        shap.summary_plot(shap_values, X_test)

        shap_plot_file = os.path.join(output_dir, "shap_summary_plot.png")
        plt.savefig(shap_plot_file)
        plt.close()
        print(f"SHAP summary plot saved to {shap_plot_file}")

    except Exception as e:
        print(f"Error in plot_shap_values: {e}")


# 9. Calibration Plot
def plot_calibration_curve(y_test, best_model, X_test, output_dir):
    """
    Plots a calibration curve to evaluate the model's probability predictions.

    Args:
    y_test: True target labels.
    best_model: Trained classification model.
    X_test: Feature data for testing.
    output_dir: Directory to save the plot.

    Returns:
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        prob_true, prob_pred = calibration_curve(
            y_test, best_model.predict_proba(X_test)[:, 1], n_bins=10
        )

        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o", label="Calibration Curve")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Probability")
        plt.title("Calibration Plot")
        plt.legend()

        plot_file_path = os.path.join(output_dir, "calibration_curve.png")
        plt.savefig(plot_file_path)
        plt.close()
    except Exception as e:
        print(f"Error in plot_calibration_curve: {e}")


# 10. Partial Dependence Plot (for NumPy arrays)
def plot_partial_dependence_plot(best_model, X_train, output_dir):
    """
    Plots partial dependence plots to show the relationship between features and predicted outcomes.
    Partial dependence plots help to visualize the effect of a feature on the model's predictions
    while holding other features constant.

    Args:
    best_model: Trained machine learning model.
    X_train: Feature data for training (pandas DataFrame).
    output_dir: Directory to save the plot.

    Returns:
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # List of feature names
        feature_names = X_train.columns

        # Plot using feature names directly
        disp = PartialDependenceDisplay.from_estimator(
            best_model,
            X_train,
            features=[0, 1, 2],  # Indices of features to plot, can be customized
            feature_names=feature_names,  # Pass feature names manually
        )

        # Saving the plot to a file
        plot_file_path = os.path.join(output_dir, "partial_dependence_plot.png")
        disp.plot()
        plt.savefig(plot_file_path)
        plt.close()
        print(f"Partial dependence plot saved to {plot_file_path}")

    except Exception as e:
        print(f"Error in plot_partial_dependence_plot: {e}")
