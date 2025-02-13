import kfp
from kfp import dsl
import pandas as pd
import mlflow

data_path = "/Users/tkax/dev/aimonetize/WIP/AIAgents/src/data/kubeflow_data.csv"
kubeflow_yaml_output_path = "/Users/tkax/dev/aimonetize/WIP/AIAgents/src/pipelines/kubeflow/basic_ml_pipeline.yaml"  # Replace with your path

# Define component functions
@dsl.component(base_image="python:3.10")
def etl_op(data_path: str) -> str:
    """ETL component: Reads data from a CSV and saves it as Parquet."""
    import pandas as pd
    import os

    output_path = "etl_data.parquet"
    df = pd.read_csv(data_path)
    df.to_parquet(output_path)
    print(f"ETL completed. Data saved to {output_path}")
    return output_path


@dsl.component(base_image="python:3.10")
def data_prep_op(input_path: str) -> str:
    """Data preparation component: Cleans and preprocesses the data."""
    import pandas as pd

    output_path = "prep_data.parquet"
    df = pd.read_parquet(input_path)
    # Basic data cleaning/preprocessing (example)
    df = df.dropna()  # Drop rows with missing values
    df = df.fillna(0)
    df.to_parquet(output_path)
    print(f"Data preparation completed. Data saved to {output_path}")
    return output_path


@dsl.component(base_image="python:3.10")
def feature_engineering_op(input_path: str) -> str:
    """Feature engineering component: Creates new features."""
    import pandas as pd

    output_path = "feature_data.parquet"
    df = pd.read_parquet(input_path)
    # Example feature engineering: Create a new feature
    df["feature_1"] = df.iloc[:, 0] * 2  # example feature
    df.to_parquet(output_path)
    print(f"Feature engineering completed. Data saved to {output_path}")
    return output_path


@dsl.component(base_image="python:3.10")
def train_model_op(input_path: str) -> str:
    """Model training component: Trains a linear regression model."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import joblib
    import mlflow

    output_path = "model.joblib"

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )  # replace with your mlflow server url
    with mlflow.start_run() as run:
        df = pd.read_parquet(input_path)
        X = df.drop(columns=df.columns[-1])
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        joblib.dump(model, output_path)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        print(f"Model training completed. Model saved to {output_path}")
    return output_path


@dsl.component(base_image="python:3.10")
def evaluate_model_op(model_path: str, input_path: str):
    """Model evaluation component: Evaluates the trained model."""
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    import joblib
    import mlflow

    with mlflow.start_run() as run:
        model = joblib.load(model_path)
        df = pd.read_parquet(input_path)
        X = df.drop(columns=df.columns[-1])
        y = df.iloc[:, -1]
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mlflow.log_metric("mse", mse)
        print(f"Model evaluation completed. MSE: {mse}")


# Define the Kubeflow Pipeline
@dsl.pipeline(
    name="Basic ML Pipeline",
    description="A basic example of a Kubeflow Pipeline for ML.",
)
def basic_ml_pipeline(
    data_path: str = data_path,
):

    etl_task = etl_op(data_path=data_path)
    data_prep_task = data_prep_op(input_path=etl_task.output)
    feature_engineering_task = feature_engineering_op(input_path=data_prep_task.output)
    train_task = train_model_op(input_path=feature_engineering_task.output)
    evaluate_task = evaluate_model_op(
        model_path=train_task.output, input_path=feature_engineering_task.output
    )


# Create a sample CSV data file
data = {
    "col1": [1, 2, 3, 4, 5],
    "col2": [6, 7, 8, 9, 10],
    "target": [11, 13, 15, 17, 19],
}
df = pd.DataFrame(data)
df.to_csv(data_path, index=False)

# Compile the pipeline
kfp.compiler.Compiler().compile(
    pipeline_func=basic_ml_pipeline, package_path=kubeflow_yaml_output_path
)

# You can upload this yaml to kubeflow pipelines
