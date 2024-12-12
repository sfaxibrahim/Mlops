from fastapi import FastAPI, File, UploadFile
import uvicorn
import pandas as pd
import io
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os  # Added for robust path handling

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)
def load_model():
    model_path = os.environ.get('MODEL_PATH', '/app/models/model.pkl')

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        print(f"Failed to load model from {model_path}: {str(e)}")
        raise


# Load the model when the module is imported
try:
    model = load_model()
except Exception as e:
    model = None


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

    columns_to_drop = ["Air_Leak", "timestamp", "Reservoirs", "COMP", "Caudal_impulses", "Pressure_switch", "H1"]
    if all(col in data.columns for col in columns_to_drop):
        data.drop(columns=columns_to_drop, inplace=True)
    else:
        return {"message": "Some required columns to drop are missing in the uploaded file!"}

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
