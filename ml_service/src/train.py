import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import mlflow
from dotenv import load_dotenv

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_pipeline', 'src'))
from db_helper import MongoDBHelper
from model import StockLSTM

load_dotenv()

# MLflow Config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Finance-LSTM-Stock-Prediction")

def load_and_preprocess(seq_length=60):
    print("Loading data from DB...")
    db = MongoDBHelper()
    ticker = os.getenv("TICKER", "BBCA.JK")
    collection_name = os.getenv("COLLECTION_NAME", "stock_data")
    data = db.get_all_data(collection_name, ticker)
    db.close()
    
    if not data:
        raise ValueError("No data available in DB for training.")
        
    df = pd.DataFrame(data)
    # We will use 'Close' price for prediction
    data = df['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    # Reshape for LSTM: (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split train and test (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

def train_model():
    seq_length = 60
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess(seq_length)
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    model = StockLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting MLflow run...")
    with mlflow.start_run():
        mlflow.log_params({
            "seq_length": seq_length,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "model_type": "LSTM"
        })
        
        print("Training model...")
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                mlflow.log_metric("train_loss", loss.item(), step=epoch)
                
        # Simple evaluation on test set
        model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            test_preds = model(X_test_t)
            test_loss = criterion(test_preds, y_test_t)
            print(f'Test MSE Loss: {test_loss.item():.4f}')
            mlflow.log_metric("test_loss", test_loss.item())
            
        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/lstm_model.pth')
        # Log model to mlflow
        mlflow.pytorch.log_model(model, "lstm_model")
        print("Model trained and saved.")

if __name__ == "__main__":
    train_model()
