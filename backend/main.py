from fastapi import FastAPI,File, UploadFile
import uvicorn
import pandas as pd
import io
from fastapi.middleware.cors import CORSMiddleware
import pickle



app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Path to the model
model_path = "../models/model.pkl"
model = None  # Placeholder for the loaded model

# Load the model when the application starts
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")


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

    columns_to_drop = ["Air_Leak","timestamp","Reservoirs", "COMP", "Caudal_impulses", "Pressure_switch", "H1"]
    if all(col in data.columns for col in columns_to_drop):
        data.drop(columns=columns_to_drop, inplace=True)
    else:
        return {"message": "Some required columns to drop are missing in the uploaded file!"}


    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
      
    

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



