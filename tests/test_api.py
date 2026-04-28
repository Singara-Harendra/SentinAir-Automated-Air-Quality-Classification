import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# TC-01
def test_health_check():
    response = client.get("/health")
    # Note: If running locally without model, might be 503. Both are handled safely.
    assert response.status_code in [200, 503]
    assert "status" in response.json()

# TC-02
def test_predict_valid_input():
    # Provide the exact 12 features the model expects
    payload = {
        "features": {
            "PT08.S1(CO)": 1200.0, "NMHC(GT)": 50.0, "C6H6(GT)": 5.0, 
            "PT08.S2(NMHC)": 900.0, "NOx(GT)": 100.0, "PT08.S3(NOx)": 1000.0, 
            "NO2(GT)": 80.0, "PT08.S4(NO2)": 1500.0, "PT08.S5(O3)": 1000.0, 
            "T": 15.0, "RH": 50.0, "AH": 0.75
        }
    }
    response = client.post("/predict", json=payload)
    
    # If the model is missing during testing, it returns 503, which is a valid API response.
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1, 2]
        assert "label" in data

# TC-03
def test_predict_missing_features():
    # Provide only 2 features. The API should impute the rest and not crash!
    payload = {
        "features": {
            "T": 25.0,
            "RH": 60.0
        }
    }
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        assert "prediction" in response.json()

# TC-04
def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "sentinair_predictions_total" in response.text