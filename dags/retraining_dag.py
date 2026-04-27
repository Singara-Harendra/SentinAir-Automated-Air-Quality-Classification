"""
Retraining DAG for SentinAir — Air Quality MLOps Pipeline

Multi-node pipeline:
  check_feedback_volume → check_error_rate → detect_data_drift
        → decide_retraining → merge_feedback_into_raw
        → preprocess_retrain_data → train_retrain_models
        → clear_feedback_table
"""

import sys
sys.path.insert(0, '/opt/airflow')

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'retraining_pipeline',
    default_args=default_args,
    description='Multi-node retraining pipeline: drift + error-rate triggered',
    schedule=timedelta(hours=1),
    catchup=False,
    tags=['retraining', 'mlops', 'sentinair'],
)

# ── Node 1: Check how many feedback rows exist ─────────────────────────────────
def check_feedback_volume(**context):
    """Push feedback count to XCom so downstream nodes can gate on it."""
    from sqlalchemy import create_engine, text
    import os
    url = os.getenv('DATABASE_URL', 'postgresql://airflow:airflow@postgres/airflow')
    engine = create_engine(url)
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT COUNT(*) FROM feedback WHERE actual IS NOT NULL"
        ))
        count = result.scalar() or 0
    context['ti'].xcom_push(key='feedback_count', value=int(count))
    print(f"[feedback_volume] {count} labelled feedback rows found")
    return int(count)

# ── Node 2: Compute error rate from feedback ───────────────────────────────────
def check_error_rate(**context):
    """Compute error rate; push to XCom."""
    from scripts.retraining import check_error_rate as _check
    rate = _check()
    context['ti'].xcom_push(key='error_rate', value=rate)
    print(f"[error_rate] current error rate = {rate:.4f} ({rate*100:.1f}%)")
    return rate

# ── Node 3: Detect data drift ──────────────────────────────────────────────────
def detect_data_drift(**context):
    """Run drift detection; push drift flag and details to XCom."""
    from scripts.retraining import load_feedback_data, load_baseline_stats, detect_data_drift as _detect
    feedback_df = load_feedback_data()
    baseline = load_baseline_stats()
    drift_flag, drift_details = _detect(feedback_df, baseline)
    context['ti'].xcom_push(key='drift_detected', value=drift_flag)
    context['ti'].xcom_push(key='drift_details', value=str(drift_details))
    print(f"[drift_detection] drift_detected={drift_flag}")
    if drift_flag:
        print(f"[drift_detection] drifted features: {list(drift_details.keys())}")
    return drift_flag

# ── Node 4: Branch — retrain or skip ──────────────────────────────────────────
def decide_retraining(**context):
    """Branch: proceed to retraining if error_rate > 5% OR drift detected."""
    ti = context['ti']
    error_rate   = ti.xcom_pull(task_ids='check_error_rate',   key='error_rate')   or 0.0
    drift_flag   = ti.xcom_pull(task_ids='detect_data_drift',  key='drift_detected') or False
    feedback_cnt = ti.xcom_pull(task_ids='check_feedback_volume', key='feedback_count') or 0

    print(f"[decide] error_rate={error_rate:.4f}, drift={drift_flag}, feedback_rows={feedback_cnt}")

    if (error_rate > 0.05 or drift_flag) and feedback_cnt > 0:
        print("[decide] → PROCEEDING with retraining")
        return 'merge_feedback_into_raw'
    print("[decide] → SKIPPING retraining")
    return 'skip_retraining'

# ── Node 5: Merge feedback into raw dataset ────────────────────────────────────
def merge_feedback_into_raw(**context):
    """
    Union old raw data + feedback data → new raw data.
    Also saves a snapshot named after the retrain round.
    """
    import os, json
    from pathlib import Path
    import pandas as pd
    from sqlalchemy import create_engine, text

    RAW_PATH    = Path('/opt/airflow/data/raw/data.csv')
    SNAP_DIR    = Path('/opt/airflow/data/processed/retrain_snapshots')
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    # Determine retrain round
    round_file = Path('/opt/airflow/artifacts/training_round.txt')
    round_num  = int(round_file.read_text().strip()) + 1 if round_file.exists() else 1
    context['ti'].xcom_push(key='retrain_round', value=round_num)

    # Load existing raw
    raw_df = pd.read_csv(RAW_PATH)
    print(f"[merge] raw dataset before merge: {raw_df.shape}")

    # Load feedback from DB
    url = os.getenv('DATABASE_URL', 'postgresql://airflow:airflow@postgres/airflow')
    engine = create_engine(url)
    query  = 'SELECT prediction, actual, features FROM feedback WHERE actual IS NOT NULL'
    fb_df  = pd.read_sql(query, engine)

    if fb_df.empty:
        print("[merge] No feedback rows — skipping merge")
        context['ti'].xcom_push(key='combined_path', value=str(RAW_PATH))
        return

    # Expand JSON features
    fb_df['features'] = fb_df['features'].apply(json.loads)
    feat_df = pd.DataFrame(list(fb_df['features']))
    feat_df['Target'] = fb_df['actual'].map({0: 'Good', 1: 'Moderate', 2: 'Poor'})

    # Union
    combined = pd.concat([raw_df, feat_df], ignore_index=True).drop_duplicates()
    print(f"[merge] combined dataset: {combined.shape}")

    # Overwrite raw with union
    combined.to_csv(RAW_PATH, index=False)

    # Save named snapshot: retrain1_raw.csv, retrain2_raw.csv ...
    snap_path = SNAP_DIR / f'retrain{round_num}_raw.csv'
    combined.to_csv(snap_path, index=False)
    print(f"[merge] snapshot saved → {snap_path}")

    context['ti'].xcom_push(key='combined_path', value=str(RAW_PATH))
    context['ti'].xcom_push(key='retrain_round', value=round_num)

# ── Node 6: Preprocess the retrain data ───────────────────────────────────────
def preprocess_retrain_data(**context):
    """Split, encode, fit preprocessor on the merged raw dataset."""
    from pathlib import Path
    from scripts.preprocessing import (
        split_train_test, feature_selection, label_encoding, create_preprocessor
    )

    ti         = context['ti']
    round_num  = ti.xcom_pull(task_ids='merge_feedback_into_raw', key='retrain_round') or 1
    snap_dir   = Path('/opt/airflow/data/processed/retrain_snapshots')
    snap_dir.mkdir(parents=True, exist_ok=True)

    base = '/opt/airflow/data/processed/intermediate'

    # Named paths per retrain round (retrain1_train.csv, etc.)
    train_raw   = f'{base}/retrain{round_num}_train_data.csv'
    test_raw    = f'{base}/retrain{round_num}_test_data.csv'
    train_clean = f'{base}/retrain{round_num}_train_clean.csv'
    train_enc   = f'{base}/retrain{round_num}_train_encoded.csv'
    test_enc    = f'{base}/retrain{round_num}_test_encoded.csv'

    split_train_test(
        input_path='/opt/airflow/data/raw/data.csv',
        train_path=train_raw,
        test_path=test_raw,
    )
    feature_selection(
        input_path=train_raw,
        output_path=train_clean,
        features_to_drop=[],
    )
    label_encoding(
        input_path=train_clean,
        output_path=train_enc,
        label_mapping_path='/opt/airflow/artifacts/label_mapping.csv',
    )
    label_encoding(
        input_path=test_raw,
        output_path=test_enc,
        label_mapping_path='/opt/airflow/artifacts/label_mapping.csv',
    )
    create_preprocessor(
        input_path=train_enc,
        output_path='/opt/airflow/artifacts/fitted_preprocessor.pkl',
    )

    ti.xcom_push(key='train_enc', value=train_enc)
    ti.xcom_push(key='test_enc',  value=test_enc)
    print(f"[preprocess] retrain{round_num} data ready → {train_enc}, {test_enc}")

# ── Node 7: Train models on merged dataset ─────────────────────────────────────
def train_retrain_models(**context):
    """Re-train all models; MLflow run names include retrain round number."""
    from scripts.model_training import train_and_log_models
    ti        = context['ti']
    round_num = ti.xcom_pull(task_ids='merge_feedback_into_raw', key='retrain_round') or 1
    train_enc = ti.xcom_pull(task_ids='preprocess_retrain_data', key='train_enc')
    test_enc  = ti.xcom_pull(task_ids='preprocess_retrain_data', key='test_enc')

    train_and_log_models(
        train_path=train_enc,
        test_path=test_enc,
        experiment_name='Air_Quality_Classification',
        round_num=round_num,          # produces run names like RandomForest_retrain1
    )
    print(f"[train] retrain round {round_num} complete")

# ── Node 8: Clear feedback table ──────────────────────────────────────────────
def clear_feedback_table(**context):
    """
    After successful retraining, truncate the feedback table so
    the next retraining cycle starts with a clean slate.
    """
    import os
    from sqlalchemy import create_engine, text
    url = os.getenv('DATABASE_URL', 'postgresql://airflow:airflow@postgres/airflow')
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text('TRUNCATE TABLE feedback'))
    print("[clear_feedback] feedback table cleared")

# ── Wire up tasks ──────────────────────────────────────────────────────────────

t_check_volume = PythonOperator(
    task_id='check_feedback_volume',
    python_callable=check_feedback_volume,
    dag=dag,
)

t_error_rate = PythonOperator(
    task_id='check_error_rate',
    python_callable=check_error_rate,
    dag=dag,
)

t_drift = PythonOperator(
    task_id='detect_data_drift',
    python_callable=detect_data_drift,
    dag=dag,
)

t_decide = BranchPythonOperator(
    task_id='decide_retraining',
    python_callable=decide_retraining,
    dag=dag,
)

t_skip = EmptyOperator(
    task_id='skip_retraining',
    dag=dag,
)

t_merge = PythonOperator(
    task_id='merge_feedback_into_raw',
    python_callable=merge_feedback_into_raw,
    dag=dag,
)

t_preprocess = PythonOperator(
    task_id='preprocess_retrain_data',
    python_callable=preprocess_retrain_data,
    dag=dag,
)

t_train = PythonOperator(
    task_id='train_retrain_models',
    python_callable=train_retrain_models,
    dag=dag,
)

t_clear = PythonOperator(
    task_id='clear_feedback_table',
    python_callable=clear_feedback_table,
    dag=dag,
)

# DAG graph
t_check_volume >> t_error_rate >> t_drift >> t_decide
t_decide >> t_skip
t_decide >> t_merge >> t_preprocess >> t_train >> t_clear