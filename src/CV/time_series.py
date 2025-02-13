from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def time_series_CV(X_train, X_test, y_train, y_test, config, search_method="grid"):
    best_model = None
    best_params = None
    best_score = float("inf")
    best_model_type = None

    print("Performing cross-validation for time series...")

    for model_type, model_config in config["models"]["time_series"].items():
        print(f"Evaluating model: {model_type}")

        model_class = model_config["library"]
        implementation = model_config["implementation"]
        param_grid = model_config["hyperparameters"]

        # Use TimeSeriesSplit for time series data
        ts_split = TimeSeriesSplit(n_splits=5)

        if search_method == "grid":
            search = GridSearchCV(
                estimator=model_class(),
                param_grid=param_grid,
                cv=ts_split,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
            )
        elif search_method == "random":
            search = RandomizedSearchCV(
                estimator=model_class(),
                param_distributions=param_grid,
                n_iter=100,
                cv=ts_split,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
                random_state=42,
            )

        search.fit(X_train, y_train)

        # Make predictions on the test set after fitting the best model
        predictions = search.best_estimator_.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        print(f"Best Params: {search.best_params_}, MSE: {mse}, RMSE: {rmse}")

        if mse < best_score:
            best_score = mse
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_model_type = model_type

    print(f"Best time series model with MSE: {best_score} and RMSE: {rmse}")

    return best_model, best_params, best_score, best_model_type



def perform_time_series_cv(X, y, FE, config, n_splits=5):
    """
    Performs time series cross-validation, including feature engineering, model training,
    and hyperparameter tuning.

    Args:
    X (pd.DataFrame): Input features DataFrame.
    y (np.array): Target variable array.
    FE (function): Feature engineering function to apply.
    config (dict): Configuration for the feature engineering function.
    n_splits (int): Number of splits for cross-validation.

    Returns:
    None: Outputs are printed directly; modify as needed to return a summary.
    """
    # Define the cross-validation scheme
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Set up the parameter grid for RandomForestRegressor
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize RandomForestRegressor
    rf = RandomForestRegressor()

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=1)

    # Iterate over each split
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        print("X_train columns:", X_train.columns)
        print("X_val columns:", X_val.columns)
        print("y_train shape:", y_train.shape)
        print("y_val shape:", y_val.shape)
        # Apply feature engineering to the training data
        X_train_fe = FE(X_train, config)
        
        # Apply the same feature engineering to the validation data
        X_val_fe = FE(X_val, config)

        # Fit the grid search to the data
        grid_search.fit(X_train_fe, y_train)

        # Best model from grid search
        best_model = grid_search.best_estimator_

        # Predict and evaluate
        predictions = best_model.predict(X_val_fe)
        mse = mean_squared_error(y_val, predictions)
        print(f"Model trained and evaluated on fold. Best Params: {grid_search.best_params_} MSE: {mse}")
        
    return mse, best_model, grid_search.best_params_

# Example usage (assuming X, y, FE, and config are defined as before)

def timeseries_CV_FE_gridsearch(X, y, FE, config):
    tscv = TimeSeriesSplit(n_splits=5)
    # Create a FunctionTransformer
    lag_transformer = FunctionTransformer(FE(X, config))

    # Pipeline with feature engineering
    pipeline = Pipeline([
        ('feature_engineering', lag_transformer),  # Apply lag features
        ('model', XGBRegressor(objective='reg:squarederror'))
    ])

    # Define parameter grid and perform grid search as usual
    grid_search = GridSearchCV(
        pipeline,
        param_grid={
            'model__n_estimators': [50, 100],
            'model__max_depth': [3, 5],
            'model__learning_rate': [0.01, 0.1]
        },
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    grid_search.fit(X, y)

    print("Best Params:", grid_search.best_params_)