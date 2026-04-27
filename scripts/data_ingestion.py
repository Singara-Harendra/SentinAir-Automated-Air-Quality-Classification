"""
Data Ingestion Module for Air Quality MLOps Pipeline

This module handles loading and initial validation of the prepared dataset.
"""

import pandas as pd
import logging
import os
from pathlib import Path

LOG_DIR = Path('logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_DIR / 'data_ingestion.log',
    filemode='a',
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger(__name__)


def _dvc_tracking_enabled() -> bool:
    return os.getenv("ENABLE_DVC_TRACKING", "").lower() in {"1", "true", "yes"}


def _try_dvc_track(file_path: str) -> None:
    """Best-effort DVC tracking; disabled by default for Airflow containers."""
    if not _dvc_tracking_enabled():
        logger.info(f"DVC tracking skipped for {file_path}: ENABLE_DVC_TRACKING is not enabled")
        return

    import subprocess
    try:
        subprocess.run(["dvc", "add", file_path], check=True, cwd=".")
        subprocess.run(["git", "add", f"{file_path}.dvc"], check=True, cwd=".")
        subprocess.run(["git", "commit", "-m", f"Track ingested data {file_path}"], check=True, cwd=".")
        logger.info(f"Tracked {file_path} with DVC")
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
        logger.warning(f"DVC tracking skipped for {file_path}: {e}")


def ingest_data(input_path: str, output_path: str) -> None:
    """
    Load and validate the prepared dataset.

    Args:
        input_path: Path to the prepared dataset (data/raw/data.csv)
        output_path: Path to save the processed dataframe
    """
    try:
        logger.info(f"Loading data from {input_path}")

        # Load the prepared dataset
        df = pd.read_csv(input_path, sep=',', na_values=-200)

        # Basic validation
        if df.empty:
            raise ValueError("Dataset is empty")

        if 'Target' not in df.columns:
            raise ValueError("Dataset must contain 'Target' column with classification labels")

        # Log dataset info
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Target distribution: {df['Target'].value_counts().to_dict()}")

        # Check for missing values
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            logger.warning(f"Missing values found: {missing_summary[missing_summary > 0].to_dict()}")

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save processed dataframe as CSV
        df.to_csv(output_path, index=False)

        logger.info(f"Data ingestion completed. Saved to {output_path}")

        _try_dvc_track(output_path)

    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise

if __name__ == "__main__":
    # For testing
    ingest_data(
        input_path="data/raw/data.csv",
        output_path="data/processedll_data.csv"
    )
