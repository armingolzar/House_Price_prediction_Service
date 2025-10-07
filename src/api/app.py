from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
import os 


app = FastAPI(title = "🏠 House Price Prediction API")

model, s_scaler, o_encoder1, o_encoder2 = None, None, None, None


MODEL_PATH = os.getenv('MODEL_PATH', ".\\models\\house_price_prediction_model.h5")
S_SCALER_PATH = os.getenv('S_SCALER_PATH', ".\\models\\standard_scaler_numerical.pkl")
O_ENCODER1_PATH = os.getenv('O_ENCODER1_PATH', ".\\models\\ordinal_encoder_category.pkl")
O_ENCODER2_PATH = os.getenv('O_ENCODER2_PATH', ".\\models\\ordinal_encoder_ordinal.pkl")


class House_Features(BaseModel):
    area : int
    bedrooms : int
    bathrooms : int
    stories : int
    parking : int
    mainroad : str
    guestroom : str
    basement : str
    hotwaterheating : str
    airconditioning : str
    prefarea : str
    furnishingstatus : str

@app.on_event("startup")
def load_artifacts():
    global model, s_scaler, o_encoder1, o_encoder2
    try:
        print(os.getcwd())
        model = load_model(MODEL_PATH)
        s_scaler = load(S_SCALER_PATH)
        o_encoder1 = load(O_ENCODER1_PATH)
        o_encoder2 = load(O_ENCODER2_PATH)

        print("✅ Models loaded successfully")

    except Exception as e:
        model, s_scaler, o_encoder1, o_encoder2 = None, None, None, None

        print("❌ Error loading artifacts:", str(e))




@app.get('/')
def home():
    return {"message" : "Welcome to the House Price Prediction API"}




@app.post('/predict')
def predict_price(data : House_Features):
    global model, s_scaler, o_encoder1, o_encoder2

    if (model is None) or (s_scaler is None) or (o_encoder1 is None) or (o_encoder2 is None):
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:

        numerical_features = np.array([[data.area, data.bedrooms, data.bathrooms, data.stories, data.parking]])
        categorical_features = np.array([[data.mainroad, data.guestroom, data.basement, data.hotwaterheating, data.airconditioning, data.prefarea]])
        ordinal_features = np.array([[data.furnishingstatus]])

        numerical_features_scaled = s_scaler.transform(numerical_features)
        categorical_features_encoded = o_encoder1.transform(categorical_features)
        ordinal_features_encoded = o_encoder2.transform(ordinal_features)

        final_features = np.hstack((numerical_features_scaled, categorical_features_encoded))
        final_features = np.hstack((final_features, ordinal_features_encoded))

        prediction = model.predict(final_features)
        price = float(prediction[0])

        return {"predicted_price" : price}
    
    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))