from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:8080/")
mlflow.set_experiment("regression_model")


with mlflow.start_run() as run:
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {'n_estimators':100,
              'random_state':42}
    
    rf = RandomForestRegressor(**params)
    
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_params(params)

    mlflow.log_metrics({"mse": mse})

    mlflow.sklearn.log_model(
        sk_model = rf,
        artifact_path = "regression_model",
        input_example = X_train,
        registered_model_name = "regression_model_sklearn_random_forest"
    )
