import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
import numpy as np

from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_data():
    return pd.DataFrame({
        'Unit price': [10, 12, 10, 50, 52],
        'Quantity': [1, 2, 1, 5, 6],
        'Rating': [5, 4, 5, 8, 9],
        'Sales': [10.5, 25.2, 10.5, 262.5, 327.6],
        'gross income': [0.5, 1.2, 0.5, 12.5, 15.6]
    })

@patch('app.api.routes.get_data')
def test_health(mock_get_data):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch('app.api.routes.get_data')
def test_data_summary(mock_get_data, mock_data):
    mock_get_data.return_value = mock_data
    response = client.get("/api/data/summary")
    assert response.status_code == 200
    assert "shape" in response.json()
    assert response.json()["shape"] == [5, 5]

@patch('app.api.routes.get_data')
def test_evaluate_model(mock_get_data, mock_data):
    mock_get_data.return_value = mock_data
    payload = {
        "features": ["Unit price", "Quantity", "Rating"],
        "k_min": 2,
        "k_max": 3
    }
    response = client.post("/api/model/evaluate", json=payload)
    assert response.status_code == 200
    assert "evaluations" in response.json()
    assert len(response.json()["evaluations"]) == 2

@patch('app.api.routes.get_data')
def test_evaluate_model_invalid_feature(mock_get_data, mock_data):
    mock_get_data.return_value = mock_data
    payload = {
        "features": ["UnknownFeature"],
        "k_min": 2,
        "k_max": 3
    }
    response = client.post("/api/model/evaluate", json=payload)
    assert response.status_code == 422
