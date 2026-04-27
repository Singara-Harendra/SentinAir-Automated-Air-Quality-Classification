"""
Model Training Module for Air Quality MLOps Pipeline

This module handles model training, evaluation, and artifact saving using MLflow.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn

LOG_DIR = Path('logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_DIR / 'model_training.log',
    filemode='a',
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path('artifacts')
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_ROUND_FILE = ARTIFACTS_DIR / 'training_round.txt'
BASELINE_STATS_FILE = ARTIFACTS_DIR / 'baseline_stats.json'


def load_data(train_path: str, test_path: str) -> tuple:
    """Load train and test data."""
    logger.info(f"Loading train data from {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_models() -> Dict[str, Any]:
    """Define models to train."""
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=60, max_depth=10, random_state=42, n_jobs=-1
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42, max_iter=1000, solver='lbfgs'
        ),
        'SVM': SVC(probability=True, random_state=42, max_iter=1000),
    }


def get_run_name(model_name: str, round_num: int) -> str:
    """Produces names like: RandomForest_train0, RandomForest_retrain1, RandomForest_retrain2"""
    label = 'train' if round_num == 0 else f'retrain{round_num}'
    return f"{model_name}_{label}"


def extract_feature_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    numeric_cols = df.select_dtypes(include='number').columns.drop('Target', errors='ignore')
    for col in numeric_cols:
        stats[col] = {'mean': float(df[col].mean()), 'std': float(df[col].std())}
    return stats


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix',
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_learning_curve(model: Any, X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y,
        cv=3,
        scoring='accuracy',
        train_sizes=np.linspace(0.2, 1.0, 4),
        n_jobs=1,
        shuffle=True,
        random_state=42,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_scores_mean, marker='o', label='Training accuracy')
    ax.plot(train_sizes, valid_scores_mean, marker='o', label='Validation accuracy')
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_training_round(round_num: int) -> None:
    TRAINING_ROUND_FILE.write_text(str(round_num))


def load_training_round() -> int:
    if TRAINING_ROUND_FILE.exists():
        return int(TRAINING_ROUND_FILE.read_text().strip())
    return 0


def train_and_evaluate_model(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_name: str,                      # FIX #6: accept run_name so filenames match
) -> Dict[str, Any]:
    """Train and evaluate a single model while generating artifacts."""
    logger.info(f'Training {model_name}')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    weighted_avg = report['weighted avg']
    metrics = {
        'accuracy': accuracy,
        'precision': weighted_avg['precision'],
        'recall': weighted_avg['recall'],
        'f1_score': weighted_avg['f1-score'],
    }

    # FIX #6: use run_name for filenames so log_artifact paths match
    confusion_path = ARTIFACTS_DIR / f'{run_name}_confusion_matrix.png'
    learning_curve_path = ARTIFACTS_DIR / f'{run_name}_learning_curve.png'

    save_confusion_matrix(y_test, y_pred, confusion_path)
    save_learning_curve(
        model,
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        learning_curve_path,
    )

    return {
        'model': model,
        'accuracy': accuracy,
        'metrics': metrics,
        'report': report,
        'confusion_path': str(confusion_path),
        'learning_curve_path': str(learning_curve_path),
    }


def log_model_to_mlflow(
    model: Any,
    model_name: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    artifacts: Dict[str, str],
    run_name: str,
    train_path: str = "",
) -> str:
    """Log model and artifacts to MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # Tag dataset name and training round for traceability
        mlflow.set_tag("dataset_name", train_path if train_path else "unknown")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("run_name", run_name)

        for artifact_name, path in artifacts.items():
            if Path(path).exists():
                mlflow.log_artifact(path, artifact_name)
            else:
                logger.warning(f"Artifact file missing, skipping: {path}")

        model_path = ARTIFACTS_DIR / f'{run_name}_model.pkl'
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path='local_model')

        mlflow.sklearn.log_model(model, 'model')

        return mlflow.active_run().info.run_id


def train_and_log_models(
    train_path: str,
    test_path: str,
    experiment_name: str = 'Air_Quality_Classification',
    round_num: int = 0,
) -> None:
    """Main training function: trains multiple models and logs to MLflow."""
    mlflow.set_experiment(experiment_name)

    train_df, test_df = load_data(train_path, test_path)
    X_train = train_df.drop('Target', axis=1)
    y_train = train_df['Target']
    X_test = test_df.drop('Target', axis=1)
    y_test = test_df['Target']

    baseline_stats = extract_feature_stats(train_df)
    save_json(baseline_stats, BASELINE_STATS_FILE)
    logger.info(f'Saved baseline feature stats to {BASELINE_STATS_FILE}')

    models = get_models()
    results: Dict[str, Any] = {}
    best_f1_score = 0.0
    best_model_name = None
    best_run_id = None

    for model_name, model in models.items():
        run_name = get_run_name(model_name, round_num)

        result = train_and_evaluate_model(
            model, model_name, X_train, y_train, X_test, y_test, run_name
        )
        results[model_name] = result

        report_path = str(ARTIFACTS_DIR / f'{run_name}_classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(result['report'], f, indent=2)

        run_artifacts = {
            'label_mapping': 'artifacts/label_mapping.csv',
            'classification_report': report_path,
            'confusion_matrix': result['confusion_path'],     # FIX #6: use result path
            'learning_curve': result['learning_curve_path'],  # FIX #6: use result path
            'baseline_stats': str(BASELINE_STATS_FILE),
        }

        run_id = log_model_to_mlflow(
            result['model'],
            model_name,
            result['metrics'],
            getattr(model, 'get_params', lambda: {})(),
            run_artifacts,
            run_name,
            train_path=train_path,
        )

        if result['metrics']['f1_score'] > best_f1_score:
            best_f1_score = result['metrics']['f1_score']
            best_model_name = model_name
            best_run_id = run_id

        logger.info(
            f"{model_name} metrics - "
            f"Accuracy: {result['metrics']['accuracy']:.4f}, "
            f"Precision: {result['metrics']['precision']:.4f}, "
            f"Recall: {result['metrics']['recall']:.4f}, "
            f"F1: {result['metrics']['f1_score']:.4f}"
        )

    best_model = results[best_model_name]['model']
    best_model_path = ARTIFACTS_DIR / 'best_model.pkl'
    joblib.dump(best_model, best_model_path)
    logger.info(f'Best model ({best_model_name}) saved to {best_model_path}')

    if best_run_id:
        register_best_model(best_model_name, best_run_id)

    save_training_round(round_num)
    logger.info(f'Best model: {best_model_name} with F1 score {best_f1_score:.4f}')


def register_best_model(best_model_name: str, best_run_id: str) -> None:
    """Register the best model in the MLflow Model Registry."""
    try:
        model_version = mlflow.register_model(f'runs:/{best_run_id}/model', 'AirQualityModel')
        logger.info(f'Registered model version: {model_version.version}')
    except Exception as e:
        logger.warning(f'Model registration failed (non-fatal): {e}')


if __name__ == '__main__':
    train_and_log_models(
        train_path='data/processed/intermediate/train_encoded.csv',
        test_path='data/processed/intermediate/test_encoded.csv',
        round_num=0,
    )