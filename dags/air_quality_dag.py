"""
Air Quality MLOps Pipeline DAG

This DAG orchestrates the complete MLOps pipeline for air quality classification.
"""

import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# FIX #1: Ensure /opt/airflow is on the path so 'scripts' package is importable
sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'air_quality_dag',
    default_args=default_args,
    description='MLOps pipeline for air quality classification',
    schedule=timedelta(days=1),
    catchup=False,
)


def data_ingestion():
    """Ingest raw CSV data and write to processed directory."""
    from scripts.data_ingestion import ingest_data
    ingest_data(
        input_path="/opt/airflow/data/raw/data.csv",
        output_path="/opt/airflow/data/processed/intermediate/full_data.csv",
    )


def preprocessing():
    """Split, clean, encode, and fit the preprocessor."""
    from scripts.preprocessing import (
        split_train_test,
        feature_selection,
        label_encoding,
        create_preprocessor,
    )

    base = "/opt/airflow/data/processed/intermediate"
    artifacts = "/opt/airflow/artifacts"

    split_train_test(
        input_path=f"{base}/full_data.csv",
        train_path=f"{base}/train_data.csv",
        test_path=f"{base}/test_data.csv",
    )
    feature_selection(
        input_path=f"{base}/train_data.csv",
        output_path=f"{base}/train_clean.csv",
        features_to_drop=[],          # NMHC(GT) is already absent from data.csv
    )
    # FIX #5: encode train first to establish mapping, then reuse same mapping for test
    label_encoding(
        input_path=f"{base}/train_clean.csv",
        output_path=f"{base}/train_encoded.csv",
        label_mapping_path=f"{artifacts}/label_mapping.csv",
    )
    label_encoding(
        input_path=f"{base}/test_data.csv",
        output_path=f"{base}/test_encoded.csv",
        label_mapping_path=f"{artifacts}/label_mapping.csv",  # FIX #5: reuse train mapping
    )
    create_preprocessor(
        input_path=f"{base}/train_encoded.csv",
        output_path=f"{artifacts}/fitted_preprocessor.pkl",
    )


def model_training():
    """Train all models and log to MLflow."""
    from scripts.model_training import train_and_log_models
    base = "/opt/airflow/data/processed/intermediate"
    train_and_log_models(
        train_path=f"{base}/train_encoded.csv",
        test_path=f"{base}/test_encoded.csv",
    )


# Tasks
ingest_task = PythonOperator(
    task_id='data_ingestion',
    python_callable=data_ingestion,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocessing',
    python_callable=preprocessing,
    dag=dag,
)

train_task = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=dag,
)

# Dependencies
ingest_task >> preprocess_task >> train_task