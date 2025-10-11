import numpy as np
import pytest 
from src.api.app import app
from fastapi.testclient import TestClient
from unittest.mock import patch


class DummyScaler:

    def transform(self, X):

        return np.ones((len(X), len(X[0])))
    
class DummyEncoder1:
    
    def transform(self, X):
        
        return np.ones((len(X), len(X[0])))
    
class DummyEncoder2:

    def transform(self, X):

        return np.ones(len(X))
    

class DummyModel:

    def predict(self, X):

        return [9000000]



@patch('src.api.app.load_model')
@patch('src.api.app.load')
def test_predict_endpoint(mock_load, mock_load_model):

    def load_side_effect(path):

        if 'standard_scaler_numerical' in path:
            return DummyScaler()
        
        if 'ordinal_encoder_category' in path:
            return DummyEncoder1
        
        if 'ordinal_encoder_ordinal' in path:
            return DummyEncoder2
        
        else:
            return None
        
    mock_load.side_effect = load_side_effect
    mock_load_model.return_value = DummyModel()

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

