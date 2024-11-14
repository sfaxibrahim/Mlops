import mlflow
import dagshub
from load_data import load_data
from modeling import train_and_log_model
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint

# Load your data
X_train, X_test, y_train, y_test = load_data()

# Set up DagsHub tracking URI
dagshub.init(repo_owner='sfaxibrahim', repo_name='Mlops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sfaxibrahim/Mlops.mlflow")

# Hyperparameter grids
xgb_param_grid = {
    'xgb__n_estimators': randint(50, 500),
    'xgb__max_depth': randint(3, 15),
    'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'xgb__subsample': [0.5, 0.7, 1.0],
    'xgb__colsample_bytree': [0.5, 0.7, 1.0]
}

rf_param_grid = {
    'rf__n_estimators':[100,200,300],
    'rf__max_depth': [5,6,9,11,13,15],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

dt_param_grid = {
    'dt__criterion':['gini','entropy'],
    'dt__max_depth': randint(3, 15),
    'dt__min_samples_split': randint(2, 10),
    'dt__min_samples_leaf': randint(1, 4),
  
}


# Train and log models
train_and_log_model(xgb.XGBClassifier(), "xgb", xgb_param_grid, X_train, X_test, y_train, y_test)
train_and_log_model(RandomForestClassifier(), "rf", rf_param_grid, X_train, X_test, y_train, y_test)
train_and_log_model(DecisionTreeClassifier(), "dt", dt_param_grid, X_train, X_test, y_train, y_test)
