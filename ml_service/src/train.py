import os
# Prevent MKL/OpenMP thread explosion which causes "could not execute a primitive" error
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import gc
import torch
torch.set_num_threads(1) # Restrict PyTorch CPU threads to prevent memory allocation crash
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from alerting import send_email_alert

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
    
    # Save dataset to CSV temporarily so we can log it to MLflow later
    dataset_path = "training_dataset.csv"
    df.to_csv(dataset_path, index=False)
    
    # We will use 'Close' price for prediction
    data_values = df['Close'].values.reshape(-1, 1)
    
    # Free up memory
    del data
    del df
    gc.collect()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)
    
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
    
    # Free memory of numpy arrays
    del X_train, y_train
    gc.collect()
    
    # Create DataLoader for batch processing to avoid memory issues
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = StockLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting MLflow run...")
    run_name = f"LSTM_Training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    with mlflow.start_run(run_name=run_name):
        # Log the raw dataset CSV as an artifact
        if os.path.exists("training_dataset.csv"):
            mlflow.log_artifact("training_dataset.csv")
            os.remove("training_dataset.csv") # Clean up local file
            print("Logged dataset CSV to MLflow.")
            
        mlflow.log_params({
            "seq_length": seq_length,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "model_type": "LSTM"
        })
        
        print("Training model with mini-batches...")
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            if (epoch+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                
        # Enhanced evaluation on test set
        model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        with torch.no_grad():
            test_preds_scaled = model(X_test_t).numpy()
            test_actual_scaled = y_test_t.numpy()
            
        # Inverse transform back to original price
        test_preds = scaler.inverse_transform(test_preds_scaled)
        test_actual = scaler.inverse_transform(test_actual_scaled)
        
        # Calculate robust regression metrics
        mse = mean_squared_error(test_actual, test_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_actual, test_preds)
        r2 = r2_score(test_actual, test_preds)
        
        # 1. Price Accuracy (Based on MAPE - Mean Absolute Percentage Error)
        mape = np.mean(np.abs((test_actual - test_preds) / test_actual))
        price_accuracy = max(0, 100 * (1 - mape)) # Cap at 0 if massively negative
        
        # 2. Directional Accuracy (Did model predict correct UP/DOWN movement?)
        # Compare actual change (today vs yesterday) against predicted change (today's pred vs yesterday's actual)
        actual_change = test_actual[1:] - test_actual[:-1]
        pred_change = test_preds[1:] - test_actual[:-1]
        directional_correct = np.sign(actual_change) == np.sign(pred_change)
        directional_acc = np.mean(directional_correct) * 100
        
        print("\n--- Model Evaluation Metrics ---")
        print(f'Price Accuracy (1-MAPE): {price_accuracy:.2f}%')
        print(f'Directional Accuracy:  {directional_acc:.2f}%')
        print(f'MSE:  {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE:  {mae:.4f}')
        print(f'R2:   {r2:.4f}')
        
        # Log all metrics to MLflow
        mlflow.log_metrics({
            "test_price_accuracy_pct": float(price_accuracy),
            "test_directional_accuracy_pct": float(directional_acc),
            "test_mse": float(mse),
            "test_rmse": float(rmse),
            "test_mae": float(mae),
            "test_r2_score": float(r2)
        })
        
        # Generate & Log Visual Chart
        plt.figure(figsize=(12, 6))
        plt.plot(test_actual, label='Actual Price', color='blue')
        plt.plot(test_preds, label='Predicted Price', color='red', linestyle='--')
        plt.title(f"Stock Price Prediction vs Actual ({os.getenv('TICKER', 'BBCA.JK')})")
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid(True)
        
        plot_path = "prediction_chart.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log image artifact to MLflow
        mlflow.log_artifact(plot_path)
        if os.path.exists(plot_path):
            os.remove(plot_path) # clean local temp file
            
        print("Logged enhanced metrics and prediction charts to MLflow.")
            
        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/lstm_model.pth')
        # Log model to mlflow and register it
        mlflow.pytorch.log_model(
            model, 
            "lstm_model",
            registered_model_name="Finance_LSTM_Stock_Predictor"
        )
        print("Model trained, saved locally, and registered to MLflow Model Registry.")
        
        # Send Alert
        send_email_alert(
            subject="✅ Retraining Successful!",
            message_body=f"Model retraining completed successfully.\n\nRun Name: {run_name}\nPrice Accuracy: {price_accuracy:.2f}%\nMSE: {mse:.4f}\n\nThe new model version has been automatically registered to MLflow."
        )

if __name__ == "__main__":
    train_model()
