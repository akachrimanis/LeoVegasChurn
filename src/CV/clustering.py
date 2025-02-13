from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import silhouette_score


def clustering_CV(X_train, X_test, y_train, y_test, config, search_method="grid"):
    best_model = None
    best_params = None
    best_score = float("-inf")
    best_model_type = None

    print("Performing cross-validation for clustering...")

    for model_type, model_config in config["models"]["clustering"].items():
        print(f"Evaluating model: {model_type}")

        model_class = model_config["library"]
        implementation = model_config["implementation"]
        param_grid = model_config["hyperparameters"]

        if search_method == "grid":
            search = GridSearchCV(
                estimator=model_class(),
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
            )
        elif search_method == "random":
            search = RandomizedSearchCV(
                estimator=model_class(),
                param_distributions=param_grid,
                n_iter=100,
                cv=5,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
                random_state=42,
            )

        search.fit(X_train, y_train)

        predictions = search.best_estimator_.predict(X_test)
        score = silhouette_score(X_test, predictions)

        print(f"Best Params: {search.best_params_}, Silhouette Score: {score}")

        if score > best_score:
            best_score = score
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_model_type = model_type

    print(f"Best clustering model with score: {best_score}")

    return best_model, best_params, best_score, best_model_type
