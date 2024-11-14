import dagshub
import mlflow
import os
import sys
import numpy as np
# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (where src is located)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

# Now you can import the module
from src.load_data import test_data

# Your existing code goes here


experiment_name = "Air_leak_compressor" 
dagshub.init(repo_owner='sfaxibrahim', repo_name='Mlops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sfaxibrahim/Mlops.mlflow")

test_df=test_data("test_data_v3")


model_xgb = mlflow.pyfunc.load_model("runs:/f1adcf948d454cb8ae29681ae6a77c91/model_artifact")
model_tree = mlflow.pyfunc.load_model("runs:/ece21258bdcc4973adefb1dd00d1af79/model_artifact")
model_rf=mlflow.pyfunc.load_model('runs:/6498a2dda408405a92f6b2e151a2c642/model_artifact')

predictions_xgb=model_xgb.predict(test_df)
predictions_tree=model_tree.predict(test_df)
predictions_rf=model_rf.predict(test_df)
print("XGBoost Model Predictions:")
print(predictions_xgb)

print("\n random Forest  Model Predictions:")
print(predictions_rf)

print("\nDecision Tree Model Predictions:")
print(predictions_tree)

# Count unique predictions for Decision Tree model
unique, counts = np.unique(predictions_tree, return_counts=True)
prediction_counts = dict(zip(unique, counts))

# Print the counts of unique predictions
print("\nPrediction Counts for Decision Tree Model:")
print(prediction_counts)



