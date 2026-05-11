import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import List
import mlflow
from dotenv import load_dotenv

load_dotenv()

# Ensure we can import from ml_service and data_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_service', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_pipeline', 'src'))

try:
    from model import StockLSTM
    from db_helper import MongoDBHelper
except ImportError:
    print("Warning: Could not import model or db_helper. Ensure PYTHONPATH is correct.")

# Define lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    print("Starting up, loading model and scaler...")
    
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fetch some data from DB to fit the scaler
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

    # Setup MLflow credentials
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 1. Try to load from MLflow Model Registry
    try:
        model_uri = "models:/Finance_LSTM_Stock_Predictor/latest"
        print(f"Attempting to load model from MLflow Registry: {model_uri}")
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        print("Successfully loaded model from MLflow Registry.")
    except Exception as e:
        print(f"Could not load from MLflow Registry: {e}. Falling back to local file...")
        
    # 2. Fallback to local file if MLflow fetch failed
    if model is None:
        path_1 = os.path.join(os.path.dirname(__file__), '..', '..', 'ml_service', 'models', 'lstm_model.pth')
        path_2 = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lstm_model.pth')
        model_path = path_1 if os.path.exists(path_1) else path_2
        
        if os.path.exists(model_path):
            model = StockLSTM()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"Model loaded successfully from local fallback: {model_path}")
        else:
            print(f"Warning: Model file not found. Tried paths:\n - {path_1}\n - {path_2}\nPredictions will fail.")

    yield # Server runs here
    # Optional cleanup logic would go here after yield

app = FastAPI(
    title="Finance MLOps API", 
    description="API to predict BBCA.JK stock prices using LSTM",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None
SEQ_LENGTH = 60

class PredictionRequest(BaseModel):
    days: int = 1

class ForecastPoint(BaseModel):
    date: str
    price: float

class PredictionResponse(BaseModel):
    ticker: str
    predictions: List[ForecastPoint]
    last_close: float


@app.get("/")
def read_root():
    return {"message": "Welcome to Finance MLOps API. Use /predict endpoint."}

@app.get("/history")
def get_history(limit: int = 60):
    """Fetch latest N historical closing prices."""
    try:
        db = MongoDBHelper()
        ticker = os.getenv("TICKER", "BBCA.JK")
        collection_name = os.getenv("COLLECTION_NAME", "stock_data")
        # Fetch data, sorting to ensure we get latest
        data = db.get_all_data(collection_name, ticker)
        db.close()
        
        if not data:
            return {"ticker": ticker, "data": []}
            
        # Sort and limit
        df = pd.DataFrame(data)
        
        # Handle both Date object or string
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
        latest = df.tail(limit)
        
        history = []
        for _, row in latest.iterrows():
            history.append({
                "date": row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
                "close": float(row['Close'])
            })
            
        return {"ticker": ticker, "history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest = None):
    global model, scaler
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded.")
    
    days = req.days if req is not None else 1
    # Cap prediction to reasonable length to prevent extreme degradation or load time
    if days > 30:
         raise HTTPException(status_code=400, detail="Max forecast period is 30 days.")

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
        
        # Determine last date and sort properly
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
        latest_data_points = df.tail(SEQ_LENGTH)
        current_seq = latest_data_points['Close'].values.reshape(-1, 1)
        last_close = float(current_seq[-1][0])
        
        last_date = latest_data_points['Date'].iloc[-1] if 'Date' in latest_data_points.columns else pd.Timestamp.now()
        
        # Setup loop for multistep forecasting
        scaled_input = scaler.transform(current_seq)
        current_window = scaled_input.tolist()
        
        forecasts = []
        temp_date = last_date
        
        for i in range(days):
            # Prep input: shape (1, seq_length, 1)
            X = np.array(current_window[-SEQ_LENGTH:]).reshape(1, SEQ_LENGTH, 1)
            X_t = torch.tensor(X, dtype=torch.float32)
            
            # Predict
            with torch.no_grad():
                pred_scaled_val = model(X_t).numpy()[0][0]
            
            # Inverse scaling
            pred_price = float(scaler.inverse_transform([[pred_scaled_val]])[0][0])
            
            # Append predicted value back into the buffer for next iterations
            current_window.append([pred_scaled_val])
            
            # Increment Business Day (ignoring weekends roughly)
            temp_date += pd.tseries.offsets.BDay(1)
            
            forecasts.append({
                "date": temp_date.strftime('%Y-%m-%d'),
                "price": pred_price
            })

        return PredictionResponse(
            ticker=ticker,
            predictions=forecasts,
            last_close=last_close
        )
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
