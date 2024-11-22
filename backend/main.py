import dagshub
import mlflow
from fastapi import FastAPI,File, UploadFile
import uvicorn
import pandas as pd
import io

app=FastAPI()

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
    
    best_run = runs.sort_values(f"metrics.{metric}", ascending=False).iloc[0]
    
    best_run_id = best_run["run_id"]
    model_uri = f"runs:/{best_run_id}/model_artifact"
    
    return model_uri, best_run_id
# model_uri, run_id = get_best_model(experiment_name, metric="test_accuracy")
# print(f"Best model URI: {model_uri}")
# print(f"Best run ID: {run_id}")


@app.get("/")
async def root():
    return {"message": "Hello World Mlops Project"}



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for uploading a dataset as input for prediction.
    """
    if not file.filename.endswith(".csv"):
        return {"message": "Only CSV files are supported!"}
    
    contents = await file.read()
    data = pd.read_csv(io.BytesIO(contents))

    columns_to_drop = ["Air_Leak", "Reservoirs", "COMP", "Caudal_impulses", "Pressure_switch", "H1"]
    if all(col in data.columns for col in columns_to_drop):
        data.drop(columns=columns_to_drop, inplace=True)
    else:
        return {"message": "Some required columns to drop are missing in the uploaded file!"}


    try:
        model_uri, run_id = get_best_model(experiment_name, metric="test_accuracy")
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        return {"message": f"Error in retrieving the best model: {str(e)}"}

    # Perform prediction
    try:
        predictions = model.predict(data)
    except Exception as e:
        return {"message": f"Error during prediction: {str(e)}"}

    return {
        "message": "File successfully uploaded and predictions generated.",
        "predictions": predictions.tolist()  
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
    
    



    # model_xgb = mlflow.pyfunc.load_model("runs:/f1adcf948d454cb8ae29681ae6a77c91/model_artifact")
    # model_tree = mlflow.pyfunc.load_model("runs:/ece21258bdcc4973adefb1dd00d1af79/model_artifact")
    # model_rf=mlflow.pyfunc.load_model('runs:/6498a2dda408405a92f6b2e151a2c642/model_artifact')

    # # Make predictions

    # predictions_xgb=model_xgb.predict()
    # predictions_tree=model_tree.predict()
    # predictions_rf=model_rf.predict()   
    # return {"predictions_xgb": predictions_xgb,"predictions_tree": predictions_tree,"predictions_rf": predictions_rf}

# predictions_xgb=model_xgb.predict(test_df)
# predictions_tree=model_tree.predict(test_df)
# predictions_rf=model_rf.predict(test_df)
# print("XGBoost Model Predictions:")
# print(predictions_xgb)

# print("\n random Forest  Model Predictions:")
# print(predictions_rf)

# print("\nDecision Tree Model Predictions:")
# print(predictions_tree)

# unique, counts = np.unique(predictions_tree, return_counts=True)
# prediction_counts = dict(zip(unique, counts))

# print("\nPrediction Counts for Decision Tree Model:")
# print(prediction_counts)



