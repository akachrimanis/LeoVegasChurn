import importlib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
from multiprocessing import get_context
from joblib.externals.loky import get_reusable_executor
from src.CV.regression_compute_metrics import compute_regression_metrics
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    median_absolute_error,
)

# Increase worker timeout
get_reusable_executor(timeout=260)


def CV(
    X_train,
    X_test,
    y_train,
    y_test,
    config,
    model_config,
    search_method="grid",
    n_splits=5,
    optimization_metric="mse",
):
    best_model = None
    best_params = None
    best_mse = float("inf")
    best_model_type = None
    ts_split = TimeSeriesSplit(n_splits=n_splits)

    print("Performing cross-validation for regression...")
    print(model_config["models"]["regression"])
    for model_type, model_config in model_config["models"]["regression"].items():
        print(f"Evaluating model: {model_type}")

        # Dynamically import model class
        model_class = getattr(
            importlib.import_module(model_config["library"]),
            model_config["implementation"],
        )
        param_grid = model_config["hyperparameters"]

        # Check if 'normalize' is in the hyperparameters, and handle it
        if "normalize" in param_grid:
            print(f"Model {model_type} has 'normalize' parameter.")
            # Some models don't support 'normalize', like LinearRegression in sklearn
            base_model = model_class()
            if hasattr(base_model, "normalize"):
                # If the model supports normalize, proceed as is
                pass
            else:
                # Remove 'normalize' from the parameter grid if the model does not support it
                param_grid = {k: v for k, v in param_grid.items() if k != "normalize"}

        # Create the model pipeline
        # pipeline = Pipeline([("scaler", StandardScaler()), ("model", model_class())])
        pipeline = Pipeline([("model", model_class())])

        # Define the search (GridSearch or RandomizedSearch)
        search = (
            GridSearchCV(
                estimator=pipeline,
                param_grid={"model__" + k: v for k, v in param_grid.items()},
                cv=ts_split,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
                verbose=3,  # Add verbosity for progress tracking
            )
            if search_method == "grid"
            else RandomizedSearchCV(
                estimator=pipeline,
                param_distributions={"model__" + k: v for k, v in param_grid.items()},
                n_iter=100,
                cv=ts_split,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
                random_state=42,
                verbose=3,  # Add verbosity for progress tracking
            )
        )
        # Fit the model and evaluate

        with parallel_backend("threading"):
            search.fit(X_train, y_train)
        # search.fit(X_train, y_train)
        # Predict on the test set

        predictions = search.best_estimator_.predict(X_test)
        n_features = X_train.shape[1]

        # Compute all metrics
        metrics = compute_regression_metrics(y_test, predictions, n_features)

        mse = mean_squared_error(y_test, predictions)
        print(f"Best Params: {search.best_params_}, MSE: {mse}")

        if mse < best_mse:
            best_mse = mse
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_model_type = model_type

    print(f"Best regression model with MSE: {best_mse}")

    return best_model, best_params, best_mse, best_model_type


# Increase worker timeout


def CV_other_metrics(
    X_train,
    X_test,
    y_train,
    y_test,
    config,
    model_config,
    search_method="grid",
    n_splits=5,
    optimization_metric="mse",
):
    best_model = None
    best_params = None
    best_score = float(
        "inf"
    )  # Adjusted to store the best score based on the selected metric
    best_model_type = None
    ts_split = TimeSeriesSplit(n_splits=n_splits)

    # Dictionary to map optimization_metric to corresponding sklearn scoring
    scoring_dict = {
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
        "medae": "neg_median_absolute_error",
    }

    # Check if the provided optimization metric is supported
    if optimization_metric not in scoring_dict:
        raise ValueError(
            f"Optimization metric '{optimization_metric}' is not supported. Choose from: {', '.join(scoring_dict.keys())}"
        )

    print(f"Performing cross-validation for regression using {optimization_metric}...")
    print(model_config["models"]["regression"])
    for model_type, model_config in model_config["models"]["regression"].items():
        print(f"Evaluating model: {model_type}")

        # Dynamically import model class
        model_class = getattr(
            importlib.import_module(model_config["library"]),
            model_config["implementation"],
        )
        param_grid = model_config["hyperparameters"]

        # Check if 'normalize' is in the hyperparameters, and handle it
        if "normalize" in param_grid:
            print(f"Model {model_type} has 'normalize' parameter.")
            base_model = model_class()
            if hasattr(base_model, "normalize"):
                pass  # If the model supports normalize, proceed as is
            else:
                param_grid = {k: v for k, v in param_grid.items() if k != "normalize"}

        # Create the model pipeline
        pipeline = Pipeline([("model", model_class())])

        # Define the search (GridSearch or RandomizedSearch)
        search = (
            GridSearchCV(
                estimator=pipeline,
                param_grid={"model__" + k: v for k, v in param_grid.items()},
                cv=ts_split,
                n_jobs=-1,
                scoring=scoring_dict[
                    optimization_metric
                ],  # Use the selected optimization metric
                verbose=3,  # Add verbosity for progress tracking
            )
            if search_method == "grid"
            else RandomizedSearchCV(
                estimator=pipeline,
                param_distributions={"model__" + k: v for k, v in param_grid.items()},
                n_iter=100,
                cv=ts_split,
                n_jobs=-1,
                scoring=scoring_dict[
                    optimization_metric
                ],  # Use the selected optimization metric
                random_state=42,
                verbose=3,  # Add verbosity for progress tracking
            )
        )

        # Fit the model and evaluate
        with parallel_backend("threading"):
            search.fit(X_train, y_train)

        # Predict on the test set
        predictions = search.best_estimator_.predict(X_test)

        # Compute all metrics
        metrics = compute_regression_metrics(y_test, predictions, X_train.shape[1])

        # Get the score based on the selected optimization metric
        if optimization_metric == "mse":
            score = mean_squared_error(y_test, predictions)
        elif optimization_metric == "mae":
            score = mean_absolute_error(y_test, predictions)
        elif optimization_metric == "r2":
            score = r2_score(y_test, predictions)
        elif optimization_metric == "medae":
            score = median_absolute_error(y_test, predictions)

        print(
            f"Best Params: {search.best_params_}, {optimization_metric.upper()}: {score}"
        )

        # Update the best model based on the selected optimization metric
        if optimization_metric == "mse" and score < best_score:
            best_score = score
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_model_type = model_type
        elif optimization_metric != "mse" and score > best_score:
            best_score = score
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_model_type = model_type

    print(f"Best regression model based on {optimization_metric.upper()}: {best_score}")
    return best_model, best_params, best_score, best_model_type
