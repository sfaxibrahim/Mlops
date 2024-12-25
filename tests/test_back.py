import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from backend import main
import pandas as pd
# Create a test client
client = TestClient(main.app)

# Mock fixture for the model
@pytest.fixture
def mock_model():
    mock = MagicMock()
    mock.predict.return_value = [0, 1, 0, 1]  # Example mock predictions
    return mock

# Test root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World Mlops Project"}

# Test prediction endpoint with valid CSV
# def test_predict_valid_csv(mock_model):
#     main.app.dependency_overrides[main.load_model] = lambda: mock_model

#     csv_data = """Air_Leak,timestamp,Reservoirs,COMP,Caudal_impulses,Pressure_switch,H1,feature1,feature2
#     0,2023-01-01,5,3,100,1,0,0.5,0.6
#     1,2023-01-02,4,2,120,0,1,0.7,0.8
#     """
#     response = client.post(
#         "/predict",
#         files={"file": ("test.csv", csv_data, "text/csv")},
#     )
#     assert response.status_code == 200
#     assert response.json()["message"] == "File successfully uploaded and predictions generated."
#     assert response.json()["predictions"] == [0, 1, 0, 1]

# # Test prediction endpoint with invalid file type
# def test_predict_invalid_file_type():
#     response = client.post(
#         "/predict",
#         files={"file": ("test.txt", "This is a test", "text/plain")},
#     )
#     assert response.status_code == 200
#     assert response.json() == {"message": "Only CSV files are supported!"}


# # Test prediction endpoint with model error
# def test_predict_model_error(mock_model):
#     mock_model.predict.side_effect = Exception("Mock prediction error")
#     main.app.dependency_overrides[main.load_model] = lambda: mock_model

#     csv_data = """Air_Leak,timestamp,Reservoirs,COMP,Caudal_impulses,Pressure_switch,H1,feature1,feature2
#     0,2023-01-01,5,3,100,1,0,0.5,0.6
#     1,2023-01-02,4,2,120,0,1,0.7,0.8
#     """
#     response = client.post(
#         "/predict",
#         files={"file": ("test.csv", csv_data, "text/csv")},
#     )
#     assert response.status_code == 200
#     assert response.json() == {"message": "Error during prediction: Mock prediction error"}
