import dagshub
import mlflow
import os


experiment_name = "Air_leak_compressor" 
dagshub.init(repo_owner='sfaxibrahim', repo_name='Mlops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sfaxibrahim/Mlops.mlflow")


def get_best_model(experiment_name, metrics=["test_precision", "test_recall", "test_f1"]):
    """
    Finds and saves the best model based on precision, recall, and F1-score.
    
    Downloads the model directly from MLflow's model artifact path and saves it locally.
    
    Parameters:
    - experiment_name (str): Name of the MLflow experiment.
    - metrics (list): List of metrics to evaluate. Default includes precision, recall, and F1-score.
    """
    
    # Fetch the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found!")

    # Extract the experiment ID
    experiment_id = experiment.experiment_id

    # Search runs within the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Find the best run based on the specified metrics
    best_run = (
        runs.sort_values(
            [f"metrics.{metric}" for metric in metrics], ascending=False
        ).iloc[0]
    )
    
    # Extract the best run ID and model URI
    best_run_id = best_run["run_id"]
    model_uri = f"runs:/{best_run_id}/model_artifact/model.pkl"

    # Download model artifacts to a local path
    local_model_dir = "../models"
    local_model_path = os.path.join(local_model_dir, "best_model.pkl")

    try:
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=local_model_dir)
        print(f"Model downloaded and saved to: {local_model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download the model: {str(e)}")


get_best_model(experiment_name)
