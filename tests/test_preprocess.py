import pytest
import pandas as pd
from unittest.mock import patch
from src.preprocess import load_data

@pytest.fixture
def mock_data():
    # Create a mock dataframe similar to data_v1.csv
    data = {
        "Reservoirs": [1, 2, 3],
        "COMP": [0, 1, 0],
        "Caudal_impulses": [0, 1, 0],
        "Pressure_switch": [1, 0, 1],
        "H1": [10, 20, 30],
        "TP2": [5, 6, 7],
        "TP3": [8, 9, 10],
        "Oil_temperature": [40, 50, 80],
        "Oil_level": [0, 1, 0],
        "Dv_electric": [0, 1, 0],
        "DV_pressure": [7, 50, 3],
        "Motor_current": [4,6,0],
        "Towers": [0,1,0],
        "MPG": [0,1,0],
        "LPS": [0,1,0],
        "Air_Leak": [0, 1, 0],
    }
    return pd.DataFrame(data)

@patch("src.preprocess.pd.read_csv")
def test_load_data(mock_read_csv, mock_data):
    # Mock pd.read_csv to return the mock_data
    mock_read_csv.return_value = mock_data
    
    X_train, X_test, y_train, y_test = load_data()
    
    # Assertions
    assert "Reservoirs" not in X_train.columns
    assert "COMP" not in X_train.columns
    assert "Air_Leak" not in X_train.columns
    assert "Caudal_impulses" not in X_train.columns
    assert "Pressure_switch" not in X_train.columns
    assert "H1" not in X_train.columns
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert X_train.shape[1] == 10
    assert X_test.shape[0] == 1
