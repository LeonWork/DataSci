"""
src/models/train.py
-------------------
Train baseline models for customer churn prediction.

Models:
    - Logistic Regression  (interpretable baseline)
    - Random Forest        (ensemble tree baseline)
    - XGBoost              (gradient boosting baseline)

Each training run is logged to MLflow with:
    - Parameters (hyperparameters)
    - Metrics    (AUC, F1, accuracy, precision, recall)
    - Artefacts  (trained model, confusion matrix image)

Usage
-----
    from src.models.train import train_all_baselines
    results = train_all_baselines()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.models.evaluate import compute_metrics, log_confusion_matrix
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MLFLOW_TRACKING_URI = f"file://{ROOT / 'mlruns'}"
EXPERIMENT_NAME = "churn-baseline"

# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULT_LR_PARAMS: dict[str, Any] = {
    "max_iter": 1000,
    "C": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "solver": "lbfgs",
}

DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3,   # handles class imbalance (~73:27 ratio)
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


# ── Core training function ────────────────────────────────────────────────────

def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    params: dict[str, Any],
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """
    Fit a model, evaluate it, and optionally log everything to MLflow.

    Args:
        model:          Unfitted sklearn-compatible estimator.
        X_train:        Training feature matrix.
        y_train:        Training labels.
        X_test:         Test feature matrix.
        y_test:         Test labels.
        model_name:     Human-readable name used for MLflow run naming.
        params:         Hyperparameter dict (logged to MLflow).
        log_to_mlflow:  If False, skip MLflow logging (useful in tests).

    Returns:
        dict with keys: model, metrics, run_id (or None)
    """
    logger.info("Training %s …", model_name)
    model.fit(X_train, y_train)

    metrics = compute_metrics(model, X_test, y_test)
    logger.info(
        "%s → AUC=%.4f  F1=%.4f  Acc=%.4f",
        model_name,
        metrics["roc_auc"],
        metrics["f1"],
        metrics["accuracy"],
    )

    run_id = None
    if log_to_mlflow:
        run_id = _log_run(model, model_name, params, metrics, X_test, y_test)

    return {"model": model, "metrics": metrics, "run_id": run_id}


def _log_run(
    model,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> str:
    """Log a single training run to MLflow. Returns the run_id."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log the model artefact
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Log confusion matrix as an image artefact
        cm_path = log_confusion_matrix(model, X_test, y_test, model_name)
        mlflow.log_artifact(str(cm_path))

        run_id = run.info.run_id
        logger.info("MLflow run logged: %s (id=%s)", model_name, run_id)

    return run_id


# ── Convenience wrappers ──────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, Any] | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Train and log a Logistic Regression model."""
    p = {**DEFAULT_LR_PARAMS, **(params or {})}
    model = LogisticRegression(**p)
    return train_model(model, X_train, y_train, X_test, y_test,
                       "LogisticRegression", p, log_to_mlflow)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, Any] | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Train and log a Random Forest model."""
    p = {**DEFAULT_RF_PARAMS, **(params or {})}
    model = RandomForestClassifier(**p)
    return train_model(model, X_train, y_train, X_test, y_test,
                       "RandomForest", p, log_to_mlflow)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, Any] | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Train and log an XGBoost model."""
    p = {**DEFAULT_XGB_PARAMS, **(params or {})}
    # Remove non-XGBClassifier kwargs if present
    p.pop("use_label_encoder", None)
    model = XGBClassifier(**p, verbosity=0)
    return train_model(model, X_train, y_train, X_test, y_test,
                       "XGBoost", p, log_to_mlflow)


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    log_to_mlflow: bool = True,
) -> dict[str, dict]:
    """
    Train all three baseline models and return a results dict.

    Returns:
        {
            "LogisticRegression": {"model": ..., "metrics": ..., "run_id": ...},
            "RandomForest":       {"model": ..., "metrics": ..., "run_id": ...},
            "XGBoost":            {"model": ..., "metrics": ..., "run_id": ...},
        }
    """
    results = {}

    results["LogisticRegression"] = train_logistic_regression(
        X_train, y_train, X_test, y_test, log_to_mlflow=log_to_mlflow
    )
    results["RandomForest"] = train_random_forest(
        X_train, y_train, X_test, y_test, log_to_mlflow=log_to_mlflow
    )
    results["XGBoost"] = train_xgboost(
        X_train, y_train, X_test, y_test, log_to_mlflow=log_to_mlflow
    )

    # Print comparison table
    logger.info("\n%s", _format_results_table(results))
    return results


def _format_results_table(results: dict) -> str:
    header = f"{'Model':<22} {'AUC':>7} {'F1':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}"
    sep = "-" * len(header)
    rows = [header, sep]
    for name, r in results.items():
        m = r["metrics"]
        rows.append(
            f"{name:<22} {m['roc_auc']:>7.4f} {m['f1']:>7.4f} "
            f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f}"
        )
    return "\n".join(rows)
