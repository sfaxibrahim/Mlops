import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch,MagicMock
from io import BytesIO
import pandas as pd
from backend.main import app

@pytest.fixture
def mock_csv_file():
    data="""
        Features, Features2, Features3, Features4,Air_leak
        1,2,3,4,0
        2,3,4,5,1
        3,45,6,9,0
        """
    file=BytesIO(data.encode('utf8'))
    file.name="mock_data.csv"
    return file

class MockModel:
    def predict(self, data):
        return [0,1,0]

@pytest.mark.asyncio
async def test_predict(mock_csv_file):
# Mock the mlflow.pyfunc.load_model to return the MockModel
    with patch("mlflow.pyfunc.load_model") as mock_load_model:
        mock_load_model.return_value = MockModel()
<<<<<<< Tabnine <<<<<<<
    def predict(self, data):#+
        """#+
        This function is responsible for making predictions using a trained model.#+
    #+
        Parameters:#+
        data (pandas.DataFrame): A pandas DataFrame containing the features for which predictions are to be made.#+
            The DataFrame should have the same columns as the training data used to train the model.#+
    #+
        Returns:#+
        list: A list of predictions made by the model for the input data.#+
            Each element in the list corresponds to a prediction for a single data point.#+
        """#+
        # Your implementation here#+
        pass#+
>>>>>>> Tabnine >>>>>>># {"conversationId":"21bb8c06-9658-4ea3-af53-76558a2dff88","source":"instruct"}

        client =TestClient(app)
        res=client.post(
            "/predict",files={"file":("mock_data.csv",mock_csv_file,"text/csv")}
        )
        assert res.status_code == 200

        json_res=res.json()
                # Assert the response contains the expected message
        assert json_res["message"] ==  "File successfully uploaded and predictions generated"

        assert json_res["predictions"] == [0,1,0]        # Assert the predictions are correct (based on MockModel)


# Test for unsupported file type ( if the file is not a CSV)
@pytest.mark.asyncio
async def test_predict_unspported_file():
    client=TestClient(app)

    res=client.post("/predict",files={"file":("mock_data.txt",b"some_text_data","text/plain")}
    )

    assert res.status_code==400
    assert res.json()["message"] == {"message":"only csv files are supported"}


@pytest.mark.asyncio
async def test_predict_missing_columns(mock_csv_file):
    data="""
        Features,Features2,Features3,
            1,2,3,
            2,3,4,
            3,45,6
            """
    file=BytesIO(data.encode('utf8'))
    file.name="mock_data.csv"

    with patch("mlflow.pyfunc.load_model")as mock_load_model:
        mock_load_model.return_value = MockModel()
        client=TestClient(app)
        response = client.post(
            "/predict", files={"file": ("mock_data.csv", file, "text/csv")}
        )
         # Assert the response contains the error message about missing columns
        assert response.status_code == 400
        assert response.json() == {"message": "Some required columns to drop are missing in the uploaded file!"}



  
