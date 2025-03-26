import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:8080/")
mlflow.set_experiment("iris")


iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo:", accuracy)

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100) # log parameter
    mlflow.log_metric("accuracy", accuracy) # log metric
    mlflow.sklearn.log_model(model, "model_random_forest") # log model