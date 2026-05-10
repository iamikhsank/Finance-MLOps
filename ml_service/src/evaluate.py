import os
import sys
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_pipeline', 'src'))
from train import load_and_preprocess
from model import StockLSTM

load_dotenv()

def evaluate_model(threshold_loss=0.05):
    """
    Evaluates the model on the latest data.
    Returns True if model needs retraining (e.g. loss > threshold), False otherwise.
    """
    print("Evaluating latest model...")
    model_path = 'models/lstm_model.pth'
    if not os.path.exists(model_path):
        print("No model found. Retraining required.")
        return True
        
    try:
        # Load test data (which represents recent data based on split)
        _, _, X_test, y_test, _ = load_and_preprocess(seq_length=60)
        
        # Load Model
        model = StockLSTM()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        criterion = nn.MSELoss()
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        with torch.no_grad():
            preds = model(X_test_t)
            loss = criterion(preds, y_test_t).item()
            
        print(f"Current Evaluation Loss (MSE): {loss:.4f}")
        
        if loss > threshold_loss:
            print(f"Loss exceeds threshold ({threshold_loss}). Retraining required.")
            return True
        else:
            print("Model performance is satisfactory. No retraining needed.")
            return False
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return True

if __name__ == "__main__":
    needs_retrain = evaluate_model()
    print(f"Needs retrain: {needs_retrain}")
