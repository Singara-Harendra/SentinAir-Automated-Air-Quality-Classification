"""
Preprocessing module for the Air Quality pipeline.

This module handles train/test splitting, feature selection, and label encoding
for the processed dataset produced by data ingestion.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

LOG_DIR = Path('logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_DIR / 'preprocessing.log',
    filemode='a',
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger(__name__)


def _dvc_tracking_enabled() -> bool:
    return os.getenv("ENABLE_DVC_TRACKING", "").lower() in {"1", "true", "yes"}


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _save_csv(df: pd.DataFrame, path: str) -> None:
    output_dir = Path(path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_label_mapping_csv(mapping: Dict[str, int], path: str) -> None:
    output_dir = Path(path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_df = pd.DataFrame(
        [{'Target': label, 'EncodedTarget': encoded_value} for label, encoded_value in mapping.items()]
    )
    mapping_df.to_csv(path, index=False)


def clean_missing_values(df: pd.DataFrame, target_col: str = 'Target') -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Input dataframe must contain a '{target_col}' column")

    initial_shape = df.shape
    df = df.dropna(subset=[target_col])
    if df.empty:
        raise ValueError("No rows remain after dropping records with missing target values")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    logger.info(f"Cleaned missing values: {initial_shape} -> {df.shape}")
    return df


def split_train_test(
    input_path: str,
    train_path: str,
    test_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Split the full dataset into train and test sets."""
    logger.info(f"Loading full dataset from {input_path}")
    df = _load_csv(input_path)

    if df.empty:
        raise ValueError("Input dataframe is empty")

    df = clean_missing_values(df, target_col='Target')

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Target'] if df['Target'].nunique() > 1 else None,
    )

    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    _save_csv(train_df, train_path)
    _save_csv(test_df, test_path)
    logger.info(f"Saved train data to {train_path} and test data to {test_path}")

    _try_dvc_track(train_path)
    _try_dvc_track(test_path)


def feature_selection(
    input_path: str,
    output_path: str,
    features_to_drop: Optional[List[str]] = None,
) -> None:
    """Drop unwanted columns from the training dataset."""
    logger.info(f"Loading dataset for feature selection from {input_path}")
    df = _load_csv(input_path)

    if features_to_drop:
        missing = [feat for feat in features_to_drop if feat not in df.columns]
        if missing:
            logger.warning(f"Requested drop features not found in dataframe (skipping): {missing}")
        df = df.drop(columns=[feat for feat in features_to_drop if feat in df.columns])

    logger.info(f"Resulting dataset shape after selection: {df.shape}")
    _save_csv(df, output_path)
    logger.info(f"Saved cleaned dataset to {output_path}")


def label_encoding(
    input_path: str,
    output_path: str,
    label_mapping_path: Optional[str] = None,
) -> None:
    """
    Encode the target labels.

    FIX #5: If label_mapping_path already exists (written by the train split),
    we LOAD and REUSE that mapping instead of building a new one.  This
    guarantees train and test labels share the same integer encoding.
    """
    logger.info(f"Loading dataset for label encoding from {input_path}")
    df = _load_csv(input_path)

    if 'Target' not in df.columns:
        raise ValueError("Input dataframe must contain a 'Target' column")

    labels = df['Target'].astype(str)

    mapping: Dict[str, int]
    if label_mapping_path and Path(label_mapping_path).exists():
        # Reuse an existing mapping (e.g. when encoding the test split)
        mapping_df = pd.read_csv(label_mapping_path)
        mapping = dict(zip(mapping_df['Target'].astype(str), mapping_df['EncodedTarget']))
        logger.info(f"Loaded existing label mapping from {label_mapping_path}: {mapping}")
    else:
        # Build a new mapping (first call, on training data)
        mapping = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
        logger.info(f"Built new label mapping: {mapping}")
        if label_mapping_path:
            _save_label_mapping_csv(mapping, label_mapping_path)
            logger.info(f"Saved label mapping to {label_mapping_path}")

    unmapped = set(labels.unique()) - set(mapping.keys())
    if unmapped:
        logger.warning(f"Labels in data not found in mapping (will become NaN): {unmapped}")

    df['Target'] = labels.map(mapping)
    _save_csv(df, output_path)
    logger.info(f"Saved encoded dataset to {output_path}")
    _try_dvc_track(output_path)


def create_preprocessor(input_path: str, output_path: str) -> None:
    """Create and fit a sklearn preprocessing pipeline (impute + scale)."""
    logger.info(f"Loading dataset for preprocessor from {input_path}")
    df = _load_csv(input_path)

    if 'Target' not in df.columns:
        raise ValueError("Input dataframe must contain a 'Target' column")

    X = df.drop('Target', axis=1)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    logger.info("Fitting preprocessor pipeline")
    pipeline.fit(X)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info(f"Saved fitted preprocessor to {output_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _try_dvc_track(file_path: str) -> None:
    """Best-effort DVC tracking; disabled by default for Airflow containers."""
    if not _dvc_tracking_enabled():
        logger.info(f"DVC tracking skipped for {file_path}: ENABLE_DVC_TRACKING is not enabled")
        return

    import subprocess
    try:
        subprocess.run(["dvc", "add", file_path], check=True, cwd=".")
        subprocess.run(["git", "add", f"{file_path}.dvc"], check=True, cwd=".")
        subprocess.run(["git", "commit", "-m", f"Track {file_path}"], check=True, cwd=".")
        logger.info(f"Tracked {file_path} with DVC")
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
        logger.warning(f"DVC tracking skipped for {file_path}: {e}")


if __name__ == '__main__':
    split_train_test(
        input_path='data/processed/intermediate/full_data.csv',
        train_path='data/processed/intermediate/train_data.csv',
        test_path='data/processed/intermediate/test_data.csv',
    )
    feature_selection(
        input_path='data/processed/intermediate/train_data.csv',
        output_path='data/processed/intermediate/train_clean.csv',
        features_to_drop=[],
    )
    label_encoding(
        input_path='data/processed/intermediate/train_clean.csv',
        output_path='data/processed/intermediate/train_encoded.csv',
        label_mapping_path='artifacts/label_mapping.csv',
    )
    label_encoding(
        input_path='data/processed/intermediate/test_data.csv',
        output_path='data/processed/intermediate/test_encoded.csv',
        label_mapping_path='artifacts/label_mapping.csv',
    )
    create_preprocessor(
        input_path='data/processed/intermediate/train_encoded.csv',
        output_path='artifacts/fitted_preprocessor.pkl',
    )
