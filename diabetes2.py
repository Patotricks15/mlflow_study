import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import itertools

# Configurar o servidor e experimento do MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080/")
mlflow.set_experiment("Diabetes")


# Carregar os dados
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target, random_state=42)

# Definir o grid de hiperparâmetros
params_grid = {
    "n_estimators": [50, 100],
    "max_depth": [4, 6],
    "max_features": [2, 4]
}

# Iterar pelas combinações dos hiperparâmetros
for n_estimators, max_depth, max_features in itertools.product(
    params_grid["n_estimators"], params_grid["max_depth"], params_grid["max_features"]
):
    with mlflow.start_run():
        # Logar os parâmetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_features", max_features)
        
        # Criar e treinar o modelo
        rf = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            max_features=max_features,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # Fazer previsões e calcular o erro
        predictions = rf.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric("mse", mse)
        
        # Logar o modelo treinado
        mlflow.sklearn.log_model(rf, "model")
        
        print(f"Run: n_estimators={n_estimators}, max_depth={max_depth}, max_features={max_features}, mse={mse}")
