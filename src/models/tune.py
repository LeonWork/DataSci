"""
src/models/tune.py
------------------
Hyperparameter optimisation for the churn XGBoost model using Optuna.

The study maximises ROC-AUC on the validation split.
Best params are logged back to MLflow as a dedicated "tuned" run.

Usage
-----
    from src.models.tune import run_study, get_best_model

    study = run_study(X_train, y_train, n_trials=50)
    model = get_best_model(study, X_train, y_train)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.models.evaluate import compute_metrics, log_confusion_matrix
from src.utils.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = f"file://{ROOT / 'mlruns'}"
EXPERIMENT_NAME = "churn-baseline"

# ── Search space ──────────────────────────────────────────────────────────────
# Ranges chosen to be meaningful for ~5 k-row datasets
PARAM_SPACE = {
    "n_estimators":      (100, 600),
    "max_depth":         (3, 9),
    "learning_rate":     (0.01, 0.3),
    "subsample":         (0.5, 1.0),
    "colsample_bytree":  (0.5, 1.0),
    "min_child_weight":  (1, 10),
    "gamma":             (0.0, 5.0),
    "reg_alpha":         (0.0, 2.0),
    "reg_lambda":        (0.5, 5.0),
    "scale_pos_weight":  (1, 6),
}

CV_FOLDS = 5
RANDOM_STATE = 42


# ── Objective ─────────────────────────────────────────────────────────────────

def _objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> float:
    """
    Optuna objective function.

    Runs 5-fold stratified CV and returns mean ROC-AUC.
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators",     *PARAM_SPACE["n_estimators"]),
        "max_depth":         trial.suggest_int("max_depth",         *PARAM_SPACE["max_depth"]),
        "learning_rate":     trial.suggest_float("learning_rate",   *PARAM_SPACE["learning_rate"], log=True),
        "subsample":         trial.suggest_float("subsample",       *PARAM_SPACE["subsample"]),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", *PARAM_SPACE["colsample_bytree"]),
        "min_child_weight":  trial.suggest_int("min_child_weight",  *PARAM_SPACE["min_child_weight"]),
        "gamma":             trial.suggest_float("gamma",           *PARAM_SPACE["gamma"]),
        "reg_alpha":         trial.suggest_float("reg_alpha",       *PARAM_SPACE["reg_alpha"]),
        "reg_lambda":        trial.suggest_float("reg_lambda",      *PARAM_SPACE["reg_lambda"]),
        "scale_pos_weight":  trial.suggest_int("scale_pos_weight",  *PARAM_SPACE["scale_pos_weight"]),
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbosity":         0,
        "eval_metric":       "logloss",
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )

    return float(scores.mean())


# ── Public API ────────────────────────────────────────────────────────────────

def run_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    study_name: str = "xgb-churn-tuning",
    show_progress: bool = True,
) -> optuna.Study:
    """
    Run an Optuna hyperparameter search over the XGBoost param space.

    Args:
        X_train:       Training features (transformed numpy array).
        y_train:       Binary training labels.
        n_trials:      Number of Optuna trials (more = better, slower).
        study_name:    Name for the Optuna study.
        show_progress: Show a tqdm progress bar if True.

    Returns:
        Completed optuna.Study object.
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
    )

    logger.info("Starting Optuna study '%s' — %d trials …", study_name, n_trials)

    study.optimize(
        lambda trial: _objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=show_progress,
        n_jobs=1,
    )

    logger.info(
        "Study complete — best AUC (CV): %.4f | params: %s",
        study.best_value,
        study.best_params,
    )
    return study


def get_best_model(
    study: optuna.Study,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBClassifier:
    """
    Retrain XGBoost on the full training set with the best params from study.

    Args:
        study:   Completed Optuna study.
        X_train: Full training feature matrix.
        y_train: Full training labels.

    Returns:
        Fitted XGBClassifier.
    """
    best_params = {
        **study.best_params,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
        "eval_metric": "logloss",
    }
    logger.info("Retraining with best params: %s", best_params)
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model


def log_tuned_model(
    model: XGBClassifier,
    study: optuna.Study,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> str:
    """
    Log the tuned model, its best params, and test metrics to MLflow.

    Returns:
        MLflow run_id string.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    metrics = compute_metrics(model, X_test, y_test)
    cm_path = log_confusion_matrix(model, X_test, y_test, "XGBoost_Tuned")

    with mlflow.start_run(run_name="XGBoost_Tuned") as run:
        mlflow.log_params({**study.best_params, "n_cv_trials": len(study.trials)})
        mlflow.log_metric("best_cv_auc", study.best_value)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(cm_path))
        run_id = run.info.run_id

    logger.info(
        "Tuned model logged — AUC=%.4f  F1=%.4f  run_id=%s",
        metrics["roc_auc"],
        metrics["f1"],
        run_id,
    )
    return run_id


def plot_optimization_history(study: optuna.Study, ax=None):
    """
    Plot the Optuna optimisation history (best AUC per trial).

    Args:
        study: Completed Optuna study.
        ax:    Optional matplotlib Axes.

    Returns:
        matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    best_so_far = [max(values[: i + 1]) for i in range(len(values))]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    ax.plot(values, alpha=0.4, color="steelblue", label="Trial AUC")
    ax.plot(best_so_far, color="crimson", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV ROC-AUC")
    ax.set_title("Optuna Optimisation History", fontweight="bold")
    ax.legend()
    return ax


def plot_param_importances(study: optuna.Study, ax=None):
    """
    Bar chart of hyperparameter importances computed by Optuna.

    Args:
        study: Completed Optuna study (needs ≥ 10 trials for stable results).
        ax:    Optional matplotlib Axes.

    Returns:
        matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    importances = optuna.importance.get_param_importances(study)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    params = list(importances.keys())
    scores = list(importances.values())
    colors = plt.cm.viridis([s / max(scores) for s in scores])

    ax.barh(params[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Hyperparameter Importances (Optuna)", fontweight="bold")
    return ax
