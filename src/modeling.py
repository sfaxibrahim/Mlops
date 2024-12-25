# train_and_log_model.py
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from imblearn.over_sampling import SMOTE
from time import time


# Function to train and log a model with MLflow
def train_and_log_model(model, model_name, param_distributions, X_train, X_test, y_train, y_test):
    # Set unique experiment name for each model
    experiment_name = "Air_leak_compressor"

    try:
        mlflow.set_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=model_name):
        start_time = time()

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Define the pipeline with scaler and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
        ])
        
        # Define scoring based on model type
        scoring = "roc_auc" if model_name == "xgb" else "accuracy"
        
        # Log basic parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_iter", 5)
        mlflow.log_param("scoring", scoring)
        mlflow.log_param("cv", 5)
        
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=5,
            scoring=scoring,
            cv=5,
            verbose=3,
            n_jobs=-1
        )
        
        # Train the model
        random_search.fit(X_train_resampled, y_train_resampled)
        elapsed_time = time() - start_time
        
        # Log best parameters and training time
        best_params = random_search.best_params_
        mlflow.log_params(best_params)
        mlflow.log_metric("training_time_sec", elapsed_time)
        
        # Get best model and evaluate
        best_model = random_search.best_estimator_
        
        
        y_pred = best_model.predict(X_test)
        
        # Metrics
        test_acc = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average="binary")
        test_recall = recall_score(y_test, y_pred, average="binary")
        test_f1 = f1_score(y_test, y_pred, average="binary")
        test_auc = roc_auc_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        
        mlflow.sklearn.log_model(best_model, "model_artifact")

       
        
        print(f"{model_name} - Test Accuracy: {test_acc:.2f}, Test AUC: {test_auc:.2f}")
