import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import List

# Ensure we can import from ml_service and data_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_service', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_pipeline', 'src'))

try:
    from model import StockLSTM
    from db_helper import MongoDBHelper
except ImportError:
    print("Warning: Could not import model or db_helper. Ensure PYTHONPATH is correct.")

app = FastAPI(title="Finance MLOps API", description="API to predict BBCA.JK stock prices using LSTM")

# Global variables for model and scaler
model = None
scaler = None
SEQ_LENGTH = 60

class PredictionResponse(BaseModel):
    predicted_price: float
    ticker: str

@app.on_event("startup")
def load_resources():
    global model, scaler
    print("Starting up, loading model and scaler...")
    
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fetch some data from DB to fit the scaler (a real app might save the scaler object with pickle)
    try:
        db = MongoDBHelper()
        ticker = os.getenv("TICKER", "BBCA.JK")
        collection_name = os.getenv("COLLECTION_NAME", "stock_data")
        data = db.get_all_data(collection_name, ticker)
        db.close()
        
        if data:
            df = pd.DataFrame(data)
            prices = df['Close'].values.reshape(-1, 1)
            scaler.fit(prices)
            print("Scaler fitted on historical data.")
    except Exception as e:
        print(f"Could not initialize scaler from DB: {e}")

    # Load Model (Check both ml_service/models and root models/ paths)
    path_1 = os.path.join(os.path.dirname(__file__), '..', '..', 'ml_service', 'models', 'lstm_model.pth')
    path_2 = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lstm_model.pth')
    
    model_path = path_1 if os.path.exists(path_1) else path_2
    
    if os.path.exists(model_path):
        model = StockLSTM()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model loaded successfully from: {model_path}")
    else:
        print(f"Warning: Model file not found. Tried paths:\n - {path_1}\n - {path_2}\nPredictions will fail.")

@app.get("/")
def read_root():
    return {"message": "Welcome to Finance MLOps API. Use /predict endpoint."}

@app.post("/predict", response_model=PredictionResponse)
def predict():
    global model, scaler
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded.")
        
    try:
        # Get latest data for prediction
        db = MongoDBHelper()
        ticker = os.getenv("TICKER", "BBCA.JK")
        collection_name = os.getenv("COLLECTION_NAME", "stock_data")
        data = db.get_all_data(collection_name, ticker)
        db.close()
        
        if len(data) < SEQ_LENGTH:
            raise HTTPException(status_code=400, detail=f"Not enough data. Need at least {SEQ_LENGTH} days.")
            
        df = pd.DataFrame(data)
        latest_prices = df['Close'].values[-SEQ_LENGTH:].reshape(-1, 1)
        
        # Scale
        scaled_prices = scaler.transform(latest_prices)
        
        # Reshape for LSTM: (1, seq_length, 1)
        X = np.reshape(scaled_prices, (1, SEQ_LENGTH, 1))
        X_t = torch.tensor(X, dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            pred_scaled = model(X_t).numpy()
            
        # Inverse transform
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        
        return PredictionResponse(predicted_price=float(pred_price), ticker=ticker)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
