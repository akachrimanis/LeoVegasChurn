import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train_model(input_path, alpha, l1_ratio):
    # Load data
    data = pd.read_csv(input_path)
    X = data.drop(columns=["target"])
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Log parameters, metrics, and model to MLflow
    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
        print("Model and metrics logged to MLflow.")


if __name__ == "__main__":
    input_path = "data/processed/engineered_data.csv"
    alpha = 0.5
    l1_ratio = 0.1
    train_model(input_path, alpha, l1_ratio)
