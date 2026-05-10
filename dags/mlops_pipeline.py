from datetime import datetime, timedelta
import sys
import os
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# Add paths for imports
sys.path.append('/opt/airflow')

# Import our custom scripts
from data_pipeline.src.ingest import run_ingestion
from ml_service.src.evaluate import evaluate_model
from ml_service.src.train import train_model

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'finance_mlops_pipeline',
    default_args=default_args,
    description='End-to-End MLOps pipeline for Stock Prediction',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

def ingest_data_task():
    print("Starting data ingestion...")
    run_ingestion()

def check_model_performance(**kwargs):
    print("Checking model performance...")
    needs_retrain = evaluate_model(threshold_loss=0.05)
    
    if needs_retrain:
        print("Model needs retraining. Branching to retrain_model_task.")
        return 'retrain_model_task'
    else:
        print("Model performance is fine. Branching to skip_retraining_task.")
        return 'skip_retraining_task'

def retrain_model_task():
    print("Retraining model...")
    train_model()

# Define tasks
ingest_task = PythonOperator(
    task_id='ingest_data_from_yfinance',
    python_callable=ingest_data_task,
    dag=dag,
)

evaluate_branch_task = BranchPythonOperator(
    task_id='evaluate_model_performance',
    python_callable=check_model_performance,
    dag=dag,
)

retrain_task = PythonOperator(
    task_id='retrain_model_task',
    python_callable=retrain_model_task,
    dag=dag,
)

skip_task = EmptyOperator(
    task_id='skip_retraining_task',
    dag=dag,
)

finish_task = EmptyOperator(
    task_id='pipeline_finished',
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# Set dependencies
ingest_task >> evaluate_branch_task
evaluate_branch_task >> [retrain_task, skip_task]
retrain_task >> finish_task
skip_task >> finish_task
