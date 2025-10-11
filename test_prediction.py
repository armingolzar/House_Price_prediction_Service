import pytest 
from src.api.app import app
from fastapi.testclient import TestClient


def test_predict_endpoint():

    with TestClient(app) as client:

# sample2 --> 10850000,7500,3,3,1,yes,no,yes,no,yes,2,yes,semi-furnished
        pyload =  {
                    "area": 7500,
                    "bedrooms": 3,
                    "bathrooms": 3,
                    "stories": 1,
                    "parking": 2,
                    "mainroad": 'yes',
                    "guestroom": 'no',
                    "basement": 'yes',
                    "hotwaterheating": 'no',
                    "airconditioning": 'yes',
                    "prefarea": 'yes',
                    "furnishingstatus": 'semi-furnished'
                    }

        response = client.post('/predict', json=pyload)
        assert response.status_code == 200
        result = response.json()
        assert 'predicted_price' in result
        assert result['predicted_price'] > 0

