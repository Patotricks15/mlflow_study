import mlflow
import statsmodels.api as sm
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:8080/")
mlflow.set_experiment("longley")

data = sm.datasets.longley.load_pandas()
df = data.data

y = df["TOTEMP"]
X = df.drop("TOTEMP", axis=1)
X = sm.add_constant(X)  

with mlflow.start_run():
    model = sm.OLS(y, X).fit()
    
    mlflow.log_param("independent_variables", X.columns.tolist())
    
    mlflow.log_metric("r_squared", model.rsquared)
    
    for var, p_val in model.pvalues.items():
        mlflow.log_metric(f"p_value_{var}", p_val)
    
    summary_str = model.summary().as_text()
    with open("model_summary.txt", "w") as f:
        f.write(summary_str)
    mlflow.log_artifact("model_summary.txt")
