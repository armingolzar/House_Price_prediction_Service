# 🏠 House Price Prediction MLOps Project

This project demonstrates a complete **MLOps pipeline** for a House Price Prediction model using **FastAPI** as the deployment framework.  
It showcases **modular code structure**, environment setup, and **practical deployment** of a trained neural network model.

---

## 🚀 Project Overview

- The model is trained on the `housing.csv` dataset containing **546 samples**:  
  - 437 samples for training  
  - 109 samples for testing  

- It predicts the **price of houses** based on various **numerical and categorical features**.

## ⚙️ Model Details
- **Model type:** Neural Network (Keras, TensorFlow)  
- **Normalization and Encoding:**  
  - `StandardScaler` for numerical features  
  - `OrdinalEncoder` for six binary categorical features  
  - `OrdinalEncoder` for one multi-category feature  
  - `MinMaxScaler` for label scaling  

- **Design:** Modular training and prediction code with `src/` folder structure for a clean **MLOps pipeline**  

---

## 📊 Model Performance
| Metric | Training | Validation |
|--------|----------|------------|
| MAE    | 0.07     | 0.09       |

- On average, each prediction has an error of about **$900,000**.  
- Considering the mean house price, this results in approximately **20% average error**.  
- The price range in the dataset is around **$11,000,000**.

---

## 🧠 Repository Structure


    house_price_prediction_service/
    │
    ├── src/
    │ ├── api/
    │ │ └── app.py # FastAPI application
    │ ├── data_loader/ # Data loading utilities
    │ ├── model/ # Model architecture and functions
    │ ├── train/ # Training scripts
    │ └── init.py
    │
    ├── models/
    │ ├── house_price_prediction_model.h5
    │ ├── standard_scaler_numerical.pkl
    │ ├── ordinal_encoder_category.pkl
    │ └── ordinal_encoder_ordinal.pkl
    │
    ├── assets/ # Plots, figures, etc.
    ├── test_request.py # Script to test FastAPI prediction endpoint
    ├── requirements.txt
    └── README.md


## 🧭 How to Run

1. Install dependencies using:



    pip install -r requirements.txt


2. Start the FastAPI service:

```
uvicorn src.api.app:app --reload
```

3. Open a new terminal in the project root and run:

```
python manul_test_request.py
```

4. You’ll receive a JSON response with the predicted house price:

``` 
Response status code: 200
Prediction: {"predicted_price": 8450000.0}
```

## 🧩 CI/CD Integration

- The project is Dockerized and ready for CI/CD pipelines.

- CI pipeline automatically:

  - Tests code

  - Builds Docker image

  - Ensures reproducibility

- CD pipeline (deployable on platforms like AWS EB, Railway, Render, Fly.io):

  - Deploys Docker container

  - Exposes a public API URL for anyone to send input JSON and receive predictions

- **Example Docker commands to run locally:**
```
docker build -t house-price-service .
docker run -p 8080:8080 house-price-service
```

## 🌐 Deployment

- The project is ready for cloud deployment:

  - Dockerized container ensures reproducible environment

  - Environment variables (MODEL_PATH, S_SCALER_PATH, etc.) allow flexible configuration

- Can be deployed to:

  - AWS Elastic Beanstalk

  - Render.com

  - Railway.app

  - Google Cloud Run

  - Fly.io


Author: Armin Golzar <br>
Field: ML Engineer <br>
Project Type: Educational / Training Demonstration