from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV

def run_logistic_regression_grid_search(
    X_train,
    y_train,
    param_grid,
    cv=5,
    scoring="average_precision",
    random_state=42,
):
    """
    Run cross-validated hyperparameter search for Logistic Regression.

    Returns:
        grid_search: Fitted GridSearchCV object.
    """
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    return grid_search