from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score


def deep_learning_CV(X_train, X_test, y_train, y_test, config, search_method="grid"):
    best_model = None
    best_params = None
    best_score = float("-inf")
    best_model_type = None

    print("Performing cross-validation for deep learning...")

    for model_type, model_config in config["models"]["deep_learning"].items():
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
                scoring="accuracy",
            )
        elif search_method == "random":
            search = RandomizedSearchCV
