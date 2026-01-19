from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load saved artifacts
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")

app = FastAPI(title="Credit Card Fraud Detection API")

# Input schema
class Transaction(BaseModel):
    features: list  # 30 features: V1–V28, Time, Amount

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict_fraud(data: Transaction):
    X = np.array(data.features).reshape(1, -1)

    # Scale Amount (last column)
    X[:, -1] = scaler.transform(X[:, [-1]]).ravel()

    prob = model.predict_proba(X)[0][1]
    prediction = int(prob >= threshold)

    return {
        "fraud_probability": round(prob, 4),
        "fraud_prediction": prediction
    }
