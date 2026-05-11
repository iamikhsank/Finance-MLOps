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
import joblib
from dotenv import load_dotenv
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

load_dotenv()

# Ensure we can import from ml_service and data_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_service', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_pipeline', 'src'))

try:
    from model import StockLSTM
    from db_helper import MongoDBHelper
except ImportError:
    print("Warning: Could not import model or db_helper. Ensure PYTHONPATH is correct.")

def load_model_and_scaler():
    global model, scaler
    print("Loading model and scaler...")
    new_model = None
    new_scaler = None
    
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model_uri = "models:/Finance_LSTM_Stock_Predictor/latest"
        print(f"Attempting to load model from MLflow Registry: {model_uri}")
        new_model = mlflow.pytorch.load_model(model_uri)
        new_model.eval()
        print("Successfully loaded model from MLflow Registry.")
        
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions("Finance_LSTM_Stock_Predictor", stages=["None"])
        if latest_versions:
            run_id = latest_versions[0].run_id
            scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
            new_scaler = joblib.load(scaler_path)
            print("Successfully loaded scaler from MLflow Artifacts.")
    except Exception as e:
        print(f"Could not load from MLflow Registry: {e}. Falling back to local file...")
        
    if new_model is None or new_scaler is None:
        path_1 = os.path.join(os.path.dirname(__file__), '..', '..', 'ml_service', 'models')
        path_2 = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        models_dir = path_1 if os.path.exists(os.path.join(path_1, 'lstm_model.pth')) else path_2
        
        model_path = os.path.join(models_dir, 'lstm_model.pth')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                new_model = StockLSTM(input_size=6)
                new_model.load_state_dict(torch.load(model_path))
                new_model.eval()
                new_scaler = joblib.load(scaler_path)
                print(f"Model and Scaler loaded successfully from local fallback: {models_dir}")
            except Exception as e2:
                 print(f"Warning: Local fallback initialization failed: {e2}")
                 return False
        else:
            print(f"Warning: Model or Scaler files not found.")
            return False
            
    if new_model is not None and new_scaler is not None:
        model = new_model
        scaler = new_scaler
        return True
    return False

# Define lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up service lifecycle...")
    load_model_and_scaler()
    yield
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


@app.post("/refresh-model")
def refresh_model():
    """Hot-reload of the model/scaler latest artifact from registry without downtime."""
    success = load_model_and_scaler()
    if success:
        return {"status": "success", "message": "Model and scaler refreshed successfully."}
    raise HTTPException(status_code=503, detail="Failed to refresh model. Check API server logs.")


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
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
        raw_closes = df['Close'].tolist()
        
        features = ['Close', 'SMA_14', 'RSI_14', 'MACD', 'BB_High', 'BB_Low']
        
        latest_data_points = df.tail(SEQ_LENGTH).copy()
        
        for f in features:
            if f not in latest_data_points.columns:
                latest_data_points[f] = 0
                
        current_seq = latest_data_points[features].values
        last_close = float(current_seq[-1][0])
        
        last_date = latest_data_points['Date'].iloc[-1] if 'Date' in latest_data_points.columns else pd.Timestamp.now()
        
        scaled_input = scaler.transform(current_seq)
        current_window = scaled_input.tolist()
        
        forecasts = []
        temp_date = last_date
        
        for i in range(days):
            X = np.array(current_window[-SEQ_LENGTH:]).reshape(1, SEQ_LENGTH, 6)
            X_t = torch.tensor(X, dtype=torch.float32)
            
            with torch.no_grad():
                pred_scaled_val = model(X_t).numpy()[0][0]
            
            dummy = np.zeros((1, 6))
            dummy[0, 0] = pred_scaled_val
            pred_price = float(scaler.inverse_transform(dummy)[0][0])
            
            raw_closes.append(pred_price)
            series_closes = pd.Series(raw_closes)
            
            new_sma = SMAIndicator(close=series_closes, window=14, fillna=True).sma_indicator().iloc[-1]
            new_rsi = RSIIndicator(close=series_closes, window=14, fillna=True).rsi().iloc[-1]
            new_macd = MACD(close=series_closes, fillna=True).macd().iloc[-1]
            
            bb = BollingerBands(close=series_closes, window=20, window_dev=2)
            new_bb_h = bb.bollinger_hband().iloc[-1]
            new_bb_l = bb.bollinger_lband().iloc[-1]
            
            new_row_unscaled = np.array([pred_price, new_sma, new_rsi, new_macd, new_bb_h, new_bb_l]).reshape(1, -1)
            new_row_scaled = scaler.transform(new_row_unscaled)[0]
            
            current_window.append(new_row_scaled.tolist())
            
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
