"""
Preprocessing module for the Air Quality pipeline.

This module handles train/test splitting, feature selection, and label encoding
for the intermediate dataset produced by data ingestion.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

LOG_DIR = Path('logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_DIR / 'preprocessing.log',
    filemode='a',
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger(__name__)


def _load_pickle(path: str) -> pd.DataFrame:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_pickle(obj: object, path: str) -> None:
    output_dir = Path(path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


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

    logger.info(
        f"Cleaned missing values: {initial_shape} -> {df.shape} after dropping missing target rows"
    )
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
    df = _load_pickle(input_path)

    if df.empty:
        raise ValueError("Input dataframe is empty")

    df = clean_missing_values(df, target_col='Target')

    logger.info("Splitting data into train and test sets")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Target'] if df['Target'].nunique() > 1 else None,
    )

    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    _save_pickle(train_df, train_path)
    _save_pickle(test_df, test_path)
    logger.info(f"Saved train data to {train_path} and test data to {test_path}")


def feature_selection(
    input_path: str,
    output_path: str,
    features_to_drop: Optional[List[str]] = None,
) -> None:
    """Drop unwanted columns from the training dataset."""
    logger.info(f"Loading dataset for feature selection from {input_path}")
    df = _load_pickle(input_path)

    if features_to_drop:
        logger.info(f"Dropping features: {features_to_drop}")
        missing = [feat for feat in features_to_drop if feat not in df.columns]
        if missing:
            logger.warning(f"Requested drop features not found: {missing}")
        df = df.drop(columns=[feat for feat in features_to_drop if feat in df.columns])

    logger.info(f"Resulting dataset shape after selection: {df.shape}")
    _save_pickle(df, output_path)
    logger.info(f"Saved cleaned dataset to {output_path}")


def label_encoding(
    input_path: str,
    output_path: str,
    label_mapping_path: Optional[str] = None,
) -> None:
    """Encode the target labels and optionally save the mapping."""
    logger.info(f"Loading dataset for label encoding from {input_path}")
    df = _load_pickle(input_path)

    if 'Target' not in df.columns:
        raise ValueError("Input dataframe must contain a 'Target' column")

    labels = df['Target'].astype(str)
    mapping: Dict[str, int] = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
    logger.info(f"Label mapping: {mapping}")

    df['Target'] = labels.map(mapping)
    _save_pickle(df, output_path)
    logger.info(f"Saved encoded dataset to {output_path}")

    if label_mapping_path:
        _save_pickle(mapping, label_mapping_path)
        logger.info(f"Saved label mapping to {label_mapping_path}")


if __name__ == '__main__':
    split_train_test(
        input_path='data/intermediate/full_data.pkl',
        train_path='data/intermediate/train_data.pkl',
        test_path='data/intermediate/test_data.pkl',
    )

    feature_selection(
        input_path='data/intermediate/train_data.pkl',
        output_path='data/intermediate/train_clean.pkl',
        features_to_drop=['NMHC(GT)'],
    )

    label_encoding(
        input_path='data/intermediate/train_clean.pkl',
        output_path='data/intermediate/train_encoded.pkl',
        label_mapping_path='artifacts/label_mapping.pkl',
    )
