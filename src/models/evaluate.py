"""
src/models/evaluate.py
----------------------
Evaluation utilities for binary classification (churn = 1).

Functions
---------
compute_metrics     → dict of AUC, F1, accuracy, precision, recall
log_confusion_matrix → saves a confusion matrix PNG and returns its path
print_report        → pretty-print sklearn classification report
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    ConfusionMatrixDisplay,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ARTEFACTS_DIR = ROOT / "models" / "artefacts"


def compute_metrics(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute standard binary classification metrics.

    Args:
        model:     Fitted sklearn-compatible classifier.
        X_test:    Test feature matrix.
        y_test:    True binary labels.
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        dict with: roc_auc, f1, accuracy, precision, recall
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc":   round(float(roc_auc_score(y_test, y_prob)),   4),
        "pr_auc":    round(float(average_precision_score(y_test, y_prob)), 4),
        "brier":     round(float(brier_score_loss(y_test, y_prob)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)),        4),
        "accuracy":  round(float(accuracy_score(y_test, y_pred)),   4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)),    4),
    }
    logger.debug("Metrics: %s", metrics)
    return metrics


def log_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> Path:
    """
    Generate and save a confusion matrix plot as a PNG.

    Args:
        model:      Fitted classifier.
        X_test:     Test feature matrix.
        y_test:     True binary labels.
        model_name: Used in filename and plot title.
        threshold:  Decision threshold.

    Returns:
        Path to saved PNG file.
    """
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = ARTEFACTS_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Confusion matrix saved to %s", out_path)
    return out_path


def print_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "",
    threshold: float = 0.5,
) -> str:
    """
    Print and return sklearn's full classification report.

    Args:
        model:      Fitted classifier.
        X_test:     Test feature matrix.
        y_test:     True binary labels.
        model_name: Displayed in the header.
        threshold:  Decision threshold.

    Returns:
        The report string.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Churn (0)", "Churn (1)"],
        digits=4,
    )
    header = f"\n{'─'*50}\n{model_name or 'Model'} Classification Report\n{'─'*50}"
    full = f"{header}\n{report}"
    print(full)
    return full


def plot_roc_curves(results: dict, ax=None) -> plt.Axes:
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
        results: Output of train_all_baselines() —
                 {model_name: {"model": ..., "metrics": ...}}
        ax:      Optional existing Axes. Creates new figure if None.

    Returns:
        The matplotlib Axes.
    """
    from sklearn.metrics import roc_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for (name, result), color in zip(results.items(), colors):
        model = result["model"]
        auc = result["metrics"]["roc_auc"]
        # roc_curve needs the test data — caller must pass it separately;
        # this function is intended for notebook use where X_test / y_test
        # are in scope. Store them on result dict in train_all_baselines if needed.
        if "X_test" in result and "y_test" in result:
            y_prob = model.predict_proba(result["X_test"])[:, 1]
            fpr, tpr, _ = roc_curve(result["y_test"], y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Baseline Models", fontweight="bold")
    ax.legend(loc="lower right")
    return ax
