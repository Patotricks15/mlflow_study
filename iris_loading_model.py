import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:8080/")
mlflow.set_experiment("iris")

model_uri = "models:/tracking-quickstart/3"

model = mlflow.sklearn.load_model(model_uri)

dados_novos = pd.DataFrame({
    "sepal length (cm)": [5.1, 6.2],
    "sepal width (cm)": [3.5, 3.4],
    "petal length (cm)": [1.4, 5.4],
    "petal width (cm)": [0.2, 2.3]
})

predicoes = model.predict(dados_novos)
print("Predições:", predicoes)
