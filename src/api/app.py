from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
import os

print("✅ app.py imported successfully")
app = FastAPI(title="🏠 House Price Prediction API")

model, s_scaler, o_encoder1, o_encoder2 = None, None, None, None


# MODEL_PATH = os.getenv("MODEL_PATH", r"./models/house_price_prediction_model.h5")
# S_SCALER_PATH = os.getenv("S_SCALER_PATH", r"./models/standard_scaler_numerical.pkl")
# O_ENCODER1_PATH = os.getenv(
#     "O_ENCODER1_PATH", r"./models/ordinal_encoder_category.pkl"
# )
# O_ENCODER2_PATH = os.getenv("O_ENCODER2_PATH", r"./models/ordinal_encoder_ordinal.pkl")


MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "house_price_prediction_model.h5"))
S_SCALER_PATH = os.getenv("S_SCALER_PATH", os.path.join("models", "standard_scaler_numerical.pkl"))
O_ENCODER1_PATH = os.getenv("O_ENCODER1_PATH", os.path.join("models", "ordinal_encoder_category.pkl"))
O_ENCODER2_PATH = os.getenv("O_ENCODER2_PATH", os.path.join("models", "ordinal_encoder_ordinal.pkl"))


class House_Features(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    prefarea: str
    furnishingstatus: str


@app.on_event("startup")
async def load_artifacts():
    import traceback
    host = "localhost"  # host for Windows browser
    port = 8080

    try:
        print("🚀 [Startup] Initializing model and scaler...")
        print(f"📁 Current working dir: {os.getcwd()}")
        print(f"📦 MODEL_PATH = {MODEL_PATH}")
        app.state.model = load_model(MODEL_PATH)
        app.state.s_scaler = load(S_SCALER_PATH)
        app.state.o_encoder1 = load(O_ENCODER1_PATH)
        app.state.o_encoder2 = load(O_ENCODER2_PATH)
        print("✅ Models loaded successfully")
        print(f"🏠 FastAPI ready! Open http://{host}:{port}/docs in your browser")

    except Exception as e:
        app.state.model = None
        app.state.s_scaler = None
        app.state.o_encoder1 = None
        app.state.o_encoder2 = None

        print("❌ Error loading artifacts:", str(e))
        traceback.print_exc()


@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}


@app.post("/predict")
def predict_price(data: House_Features):

    if (
        (app.state.model is None)
        or (app.state.s_scaler is None)
        or (app.state.o_encoder1 is None)
        or (app.state.o_encoder2 is None)
    ):
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:

        numerical_features = np.array(
            [[data.area, data.bedrooms, data.bathrooms, data.stories, data.parking]]
        )
        categorical_features = np.array(
            [
                [
                    data.mainroad,
                    data.guestroom,
                    data.basement,
                    data.hotwaterheating,
                    data.airconditioning,
                    data.prefarea,
                ]
            ]
        )
        ordinal_features = np.array([[data.furnishingstatus]])

        numerical_features_scaled = app.state.s_scaler.transform(numerical_features)
        categorical_features_encoded = app.state.o_encoder1.transform(categorical_features)
        ordinal_features_encoded = app.state.o_encoder2.transform(ordinal_features)

        final_features = np.hstack(
            (numerical_features_scaled, categorical_features_encoded)
        )
        final_features = np.hstack((final_features, ordinal_features_encoded))

        prediction = app.state.model.predict(final_features)
        price = float(prediction[0])

        return {"predicted_price": int((price * 11000000) + 175000)}

    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))
    