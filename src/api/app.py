from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title = "House Price Prediction API")

@app.get("/health")
def health():
    return {"status" : "ok"}


class PredictRequest(BaseModel):
    features: Dict[str, Any]


@app.post("/predict")
def predict(req: PredictRequest):
    return {"recieved_feature" : req.features, "predicted_price" : 123456788}