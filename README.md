


## For Dagshub

import dagshub
dagshub.init(repo_owner='sfaxibrahim', repo_name='Mlops', mlflow=True)


https://dagshub.com/sfaxibrahim/Mlops.mlflow
import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)