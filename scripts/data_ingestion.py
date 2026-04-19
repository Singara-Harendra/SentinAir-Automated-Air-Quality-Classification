"""
Data Ingestion Module for Air Quality MLOps Pipeline

This module handles loading and initial validation of the prepared dataset.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        # Save processed dataframe
        with open(output_path, 'wb') as f:
            pickle.dump(df, f)

        logger.info(f"Data ingestion completed. Saved to {output_path}")

    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise

if __name__ == "__main__":
    # For testing
    ingest_data(
        input_path="data/raw/data.csv",
        output_path="data/intermediate/full_data.pkl"
    )