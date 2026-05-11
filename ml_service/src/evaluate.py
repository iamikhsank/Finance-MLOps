import os
import sys
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
import mlflow
from scipy.stats import ks_2samp
from alerting import send_email_alert

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_pipeline', 'src'))
from train import load_and_preprocess
from model import StockLSTM

load_dotenv()

def evaluate_model(threshold_loss=0.05):
    """
    Evaluates the model on the latest data.
    Returns True if model needs retraining (e.g. loss > threshold), False otherwise.
    """
    # Setup MLflow credentials
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("Evaluating latest model...")
    model = None
    
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
        model_path = 'models/lstm_model.pth'
        if not os.path.exists(model_path):
            print("No local model found either. Retraining required.")
            return True
            
        print(f"Loading local model from {model_path}...")
        model = StockLSTM()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
    try:
        # Load test data (which represents recent data based on split)
        X_train, y_train, X_test, y_test, _ = load_and_preprocess(seq_length=60)
        
        # --- DATA DRIFT DETECTION (Statistical KS Test) ---
        print("Checking for Data Drift...")
        # We compare the training distribution (reference) against test distribution (current)
        ks_stat, p_value = ks_2samp(y_train.flatten(), y_test.flatten())
        data_drift_detected = p_value < 0.05  # 95% confidence that distribution shifted
        
        print(f"KS Statistic: {ks_stat:.4f}, P-Value: {p_value:.4f}")
        if data_drift_detected:
            print(f"WARNING: Statistical Data Drift Detected! P-Value ({p_value:.4f}) < 0.05")
        else:
            print("No Data Drift detected. Distribution is stable.")
            
        # Log drift metrics to MLflow if tracking is enabled
        if MLFLOW_TRACKING_URI:
            with mlflow.start_run(run_name="Evaluation_Drift_Check"):
                mlflow.log_metrics({
                    "drift_ks_stat": float(ks_stat),
                    "drift_p_value": float(p_value)
                })
                mlflow.log_param("data_drift_detected", str(data_drift_detected))
        # ----------------------------------------------------
        
        criterion = nn.MSELoss()
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        with torch.no_grad():
            preds = model(X_test_t)
            loss = criterion(preds, y_test_t).item()
            
        print(f"Current Evaluation Loss (MSE): {loss:.4f}")
        
        if data_drift_detected:
            print("Retraining required proactively due to Data Drift.")
            send_email_alert(
                subject="🚨 Data Drift Detected!",
                message_body=f"Data Drift has been detected in the Finance MLOps pipeline.\n\nKS Statistic: {ks_stat:.4f}\nP-Value: {p_value:.4f} < 0.05\n\nAutomated retraining has been triggered."
            )
            return True
            
        if loss > threshold_loss:
            print(f"Loss exceeds threshold ({threshold_loss}). Retraining required due to Concept Drift (Performance Decay).")
            send_email_alert(
                subject="⚠️ Performance Decay Detected!",
                message_body=f"Model performance has dropped below the acceptable threshold.\n\nMSE Loss: {loss:.4f} > {threshold_loss}\n\nAutomated retraining has been triggered."
            )
            return True
            
        print("Model performance is satisfactory. No retraining needed.")
        return False
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return True

if __name__ == "__main__":
    needs_retrain = evaluate_model()
    print(f"Needs retrain: {needs_retrain}")
