"""
Retraining Module for Air Quality MLOps Pipeline

This module handles drift detection, feedback integration, and retraining.
"""

import json
import logging
import os
import smtplib
import subprocess
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine

LOG_DIR = Path('logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_DIR / 'retraining.log',
    filemode='a',
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path('artifacts')
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_ROUND_FILE = ARTIFACTS_DIR / 'training_round.txt'
BASELINE_STATS_FILE = ARTIFACTS_DIR / 'baseline_stats.json'


def dvc_tracking_enabled() -> bool:
    return os.getenv('ENABLE_DVC_TRACKING', '').lower() in {'1', 'true', 'yes'}


def get_database_url() -> str:
    return os.getenv('DATABASE_URL', 'postgresql://airflow:airflow@postgres/airflow')


def get_email_settings() -> Dict[str, str]:
    return {
        'smtp_server': os.getenv('ALERT_SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': os.getenv('ALERT_SMTP_PORT', '587'),
        'smtp_username': os.getenv('ALERT_SMTP_USERNAME', ''),
        'smtp_password': os.getenv('ALERT_SMTP_PASSWORD', ''),
        'email_from': os.getenv('ALERT_EMAIL_FROM', ''),
        'email_to': os.getenv('ALERT_EMAIL_TO', ''),
    }


def send_email(subject: str, body: str) -> None:
    settings = get_email_settings()
    if not settings['email_to'] or not settings['email_from']:
        logger.warning('Email settings are incomplete; skipping alert email.')
        return

    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = settings['email_from']
    message['To'] = settings['email_to']

    try:
        with smtplib.SMTP(settings['smtp_server'], int(settings['smtp_port'])) as server:
            server.starttls()
            if settings['smtp_username'] and settings['smtp_password']:
                server.login(settings['smtp_username'], settings['smtp_password'])
            server.sendmail(settings['email_from'], [settings['email_to']], message.as_string())
        logger.info(f'Alert email sent: {subject}')
    except Exception as exc:
        logger.error(f'Failed to send alert email: {exc}')


def load_feedback_data() -> pd.DataFrame:
    """Load feedback data from database and deduplicate rows."""
    engine = create_engine(get_database_url())
    query = 'SELECT prediction, actual, features FROM feedback WHERE actual IS NOT NULL'
    try:
        df = pd.read_sql(query, engine)
    except Exception as exc:
        logger.error(f'Failed to load feedback data: {exc}')
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df['features'] = df['features'].apply(json.loads)
    features_df = pd.DataFrame(list(df['features']))
    combined = pd.concat([features_df.reset_index(drop=True), df[['prediction', 'actual']].reset_index(drop=True)], axis=1)
    combined['features_str'] = df['features'].apply(lambda v: json.dumps(v, sort_keys=True))
    combined = combined.drop_duplicates(subset=['features_str', 'actual'])
    combined = combined.drop(columns=['features_str', 'prediction'])
    combined = combined.rename(columns={'actual': 'Target'})
    return combined


def check_error_rate() -> float:
    """Compute error rate from feedback records."""
    engine = create_engine(get_database_url())
    query = 'SELECT prediction, actual FROM feedback WHERE actual IS NOT NULL'
    try:
        df = pd.read_sql(query, engine)
    except Exception as exc:
        logger.error(f'Failed to compute error rate: {exc}')
        return 0.0

    if df.empty:
        return 0.0

    df['prediction'] = df['prediction'].astype(int)
    df['actual'] = df['actual'].astype(int)
    return 1.0 - accuracy_score(df['actual'], df['prediction'])


def extract_feature_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    numeric_cols = df.select_dtypes(include='number').columns.drop('Target', errors='ignore')
    for col in numeric_cols:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std())
        }
    return stats


def load_baseline_stats() -> Dict[str, Any]:
    if BASELINE_STATS_FILE.exists():
        return json.loads(BASELINE_STATS_FILE.read_text())
    return {}


def detect_data_drift(feedback_df: pd.DataFrame, baseline_stats: Dict[str, Any], threshold: float = 0.10) -> Tuple[bool, Dict[str, Dict[str, float]]]:
    if feedback_df.empty or not baseline_stats:
        return False, {}

    feedback_stats = extract_feature_stats(feedback_df)
    drifted = {}

    for feature, baseline in baseline_stats.items():
        if feature not in feedback_stats:
            continue

        current = feedback_stats[feature]
        mean_change = abs(current['mean'] - baseline['mean']) / max(abs(baseline['mean']), 1e-6)
        std_change = abs(current['std'] - baseline['std']) / max(abs(baseline['std']), 1e-6)

        if mean_change > threshold or std_change > threshold:
            drifted[feature] = {
                'baseline_mean': baseline['mean'],
                'current_mean': current['mean'],
                'mean_change_pct': round(mean_change * 100, 2),
                'baseline_std': baseline['std'],
                'current_std': current['std'],
                'std_change_pct': round(std_change * 100, 2),
            }

    return bool(drifted), drifted


def save_to_dvc(file_path: str) -> None:
    """Best-effort DVC tracking; disabled by default for Airflow containers."""
    if not dvc_tracking_enabled():
        logger.info(f'DVC tracking skipped for {file_path}: ENABLE_DVC_TRACKING is not enabled')
        return

    try:
        subprocess.run(['dvc', 'add', file_path], check=True)
        subprocess.run(['git', 'add', f'{file_path}.dvc'], check=True)
        subprocess.run(['git', 'commit', '-m', f'Track processed dataset {file_path} with DVC'], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as exc:
        logger.warning(f'DVC tracking skipped for {file_path}: {exc}')


def get_retrain_round() -> int:
    if TRAINING_ROUND_FILE.exists():
        return int(TRAINING_ROUND_FILE.read_text().strip()) + 1
    return 1


def preprocess_and_train(combined_path: str, round_num: int) -> None:
    from scripts.model_training import train_and_log_models
    from scripts.preprocessing import split_train_test, feature_selection, label_encoding, create_preprocessor

    split_train_test(
        input_path=combined_path,
        train_path='data/processed/retrain_train_data.csv',
        test_path='data/processed/retrain_test_data.csv',
    )

    feature_selection(
        input_path='data/processed/retrain_train_data.csv',
        output_path='data/processed/retrain_train_clean.csv',
        features_to_drop=['NMHC(GT)'],
    )

    label_encoding(
        input_path='data/processed/retrain_train_clean.csv',
        output_path='data/processed/retrain_train_encoded.csv',
        label_mapping_path='artifacts/label_mapping.csv',
    )

    label_encoding(
        input_path='data/processed/retrain_test_data.csv',
        output_path='data/processed/retrain_test_encoded.csv',
    )

    create_preprocessor(
        input_path='data/processed/retrain_train_encoded.csv',
        output_path='artifacts/fitted_preprocessor.pkl',
    )

    # Save preprocessed datasets to DVC
    save_to_dvc('data/processed/retrain_train_encoded.csv')
    save_to_dvc('data/processed/retrain_test_encoded.csv')

    train_and_log_models(
        train_path='data/processed/retrain_train_encoded.csv',
        test_path='data/processed/retrain_test_encoded.csv',
        experiment_name='Air_Quality_Classification',
        round_num=round_num,
    )


def combine_datasets(raw_path: str) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_path)
    feedback_df = load_feedback_data()
    if feedback_df.empty:
        return raw_df

    combined_df = pd.concat([raw_df, feedback_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    logger.info(f'Combined dataset shape: {combined_df.shape}')
    # Update raw dataset with union
    combined_df.to_csv(raw_path, index=False)
    save_to_dvc(raw_path)
    return combined_df


def retrain_pipeline() -> None:
    error_rate = check_error_rate()
    feedback_df = load_feedback_data()
    baseline_stats = load_baseline_stats()

    drift_detected, drift_details = detect_data_drift(feedback_df, baseline_stats)
    logger.info(f'Error rate: {error_rate}, drift detected: {drift_detected}')

    if drift_detected:
        body = f"Data drift detected in feedback data. Details:\n{json.dumps(drift_details, indent=2)}"
        send_email('Data Drift Detected', body)

    if drift_detected or error_rate > 0.05:
        combined_df = combine_datasets('data/raw/data.csv')
        processed_path = 'data/raw/data.csv'
        retrain_round = get_retrain_round()
        preprocess_and_train(processed_path, retrain_round)

        ack_body = (
            f'Retraining round {retrain_round} completed.\n'
            f'Error rate: {error_rate}\n'
            f'Drift detected: {drift_detected}\n'
            f'Drift details: {json.dumps(drift_details, indent=2)}'
        )
        send_email('Retraining Completed', ack_body)
    else:
        logger.info('No retraining triggered; data drift not detected and error rate below threshold.')


if __name__ == '__main__':
    retrain_pipeline()
