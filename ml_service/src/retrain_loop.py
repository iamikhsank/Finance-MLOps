import sys
import os

sys.path.append(os.path.dirname(__file__))
from evaluate import evaluate_model
from train import train_model

def run_retrain_pipeline():
    """
    Pipeline to:
    1. Evaluate the current model.
    2. Retrain if necessary.
    """
    print("--- Starting Retrain Loop ---")
    needs_retrain = evaluate_model(threshold_loss=0.05)
    
    if needs_retrain:
        print(">>> Triggering Retraining Pipeline <<<")
        train_model()
    else:
        print(">>> No retraining needed at this time <<<")
    print("--- Retrain Loop Completed ---")

if __name__ == "__main__":
    run_retrain_pipeline()
