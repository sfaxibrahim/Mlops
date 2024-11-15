import mlflow
import dagshub





experiment_name = "Air_leak_compressor" 
dagshub.init(repo_owner='sfaxibrahim', repo_name='Mlops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sfaxibrahim/Mlops.mlflow")


def get_best_model(experiment_name, metric="test_accuracy"):
    # Get the experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found!")
    
    experiment_id = experiment.experiment_id
    
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    # Sort by the specified metric
    best_run = runs.sort_values(f"metrics.{metric}", ascending=False).iloc[0]
    
    # Get the best run ID and model URI
    best_run_id = best_run["run_id"]
    model_uri = f"runs:/{best_run_id}/model_artifact"
    
    return model_uri, best_run_id

# Define the experiment name

# Get the best model and its details
model_uri, run_id = get_best_model(experiment_name, metric="test_accuracy")
print(f"Best model URI: {model_uri}")
print(f"Best run ID: {run_id}")

