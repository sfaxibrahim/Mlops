## Git 
```bash
git init
git fetch 
git add .
git commit -m "msg"
git branch -M main
git remote add origin <ssh link or http link >
git push 
git pull 
git status
```
## DVC
```bash
dvc init
dvc add ./data/ ./models/
dvc remote add --default myremote gdrive://id of folder of google drive 
dvc push 
dvc status

```


## For Dagshub

import dagshub
dagshub.init(repo_owner='sfaxibrahim', repo_name='Mlops', mlflow=True)


https://dagshub.com/sfaxibrahim/Mlops.mlflow
import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

  
MLFLOW_TRACKING_URI="https://dagshub.com/sfaxibrahim/Mlops.mlflow"
MLFLOW_TRACKING_USERNAME=sfaxibrahim
MLFLOW_TRACKING_PASSWORD=token