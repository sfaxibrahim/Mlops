import mlflow

mlflow.set_tracking_uri("https://dagshub.com/sfaxibrahim/Mlops.mlflow")
print(mlflow.list_experiments())



