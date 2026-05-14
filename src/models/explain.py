"""
src/models/explain.py
---------------------
SHAP-based model explainability for the tuned XGBoost churn classifier.

Generates:
    - Summary plot      (global feature importance beeswarm)
    - Bar plot          (mean |SHAP| per feature)
    - Waterfall plot    (single-prediction explanation)
    - Dependence plot   (feature interaction visualisation)
    - Feature names     (reconstructed from the ColumnTransformer pipeline)

Usage
-----
    from src.models.explain import (
        compute_shap_values,
        plot_summary,
        plot_waterfall,
        get_feature_names,
    )
    shap_values = compute_shap_values(model, X_test)
    plot_summary(shap_values, X_test, feature_names)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.compose import ColumnTransformer

from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ARTEFACTS_DIR = ROOT / "models" / "artefacts"


# ── Feature names ──────────────────────────────────────────────────────────────

def get_feature_names(pipeline: ColumnTransformer) -> list[str]:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.

    Args:
        pipeline: Fitted sklearn ColumnTransformer (from build_pipeline()).

    Returns:
        List of feature name strings in pipeline output column order.
    """
    names: list[str] = []

    for name, transformer, cols in pipeline.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            # Works for OneHotEncoder sub-pipeline
            try:
                sub_names = transformer.get_feature_names_out()
                names.extend(sub_names.tolist())
            except Exception:
                names.extend(cols if isinstance(cols, list) else list(cols))
        else:
            names.extend(cols if isinstance(cols, list) else list(cols))

    logger.debug("Extracted %d feature names from pipeline", len(names))
    return names


# ── SHAP values ────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X: np.ndarray,
    max_samples: int = 500,
) -> shap.Explanation:
    """
    Compute SHAP values using TreeExplainer (fast for XGBoost / RF).

    Args:
        model:       Fitted tree-based classifier (XGBClassifier / RF).
        X:           Feature matrix (numpy array or DataFrame).
        max_samples: Cap the background dataset size for speed.

    Returns:
        shap.Explanation object containing SHAP values for class 1 (Churn).
    """
    logger.info("Computing SHAP values (n_samples=%d) …", min(len(X), max_samples))

    background = X[:max_samples] if len(X) > max_samples else X

    explainer = shap.TreeExplainer(model, data=background, model_output="probability")
    shap_values = explainer(X)

    # For binary classification, keep only the positive-class (Churn=1) values
    if shap_values.values.ndim == 3:
        shap_values = shap_values[..., 1]

    logger.info("SHAP values computed — shape: %s", shap_values.values.shape)
    return shap_values


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_summary(
    shap_values: shap.Explanation,
    feature_names: list[str] | None = None,
    max_display: int = 20,
    save: bool = True,
) -> None:
    """
    Beeswarm summary plot — shows direction and magnitude of each feature's impact.

    Args:
        shap_values:   Output of compute_shap_values().
        feature_names: Column name list (same order as X columns).
        max_display:   Maximum number of features to show.
        save:          If True, save PNG to models/artefacts/.
    """
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary — Feature Impact on Churn Probability",
              fontweight="bold", pad=12)
    plt.tight_layout()

    if save:
        out = ARTEFACTS_DIR / "shap_summary.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        logger.info("Saved SHAP summary plot → %s", out)

    plt.show()


def plot_bar(
    shap_values: shap.Explanation,
    feature_names: list[str] | None = None,
    max_display: int = 20,
    save: bool = True,
) -> None:
    """
    Bar chart of mean absolute SHAP values (global feature importance).

    Args:
        shap_values:   Output of compute_shap_values().
        feature_names: Column name list.
        max_display:   Maximum number of features to show.
        save:          If True, save PNG to models/artefacts/.
    """
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.bar(
        shap_values,
        max_display=max_display,
        feature_names=feature_names,
        show=False,
    )
    plt.title("Mean |SHAP Value| — Global Feature Importance",
              fontweight="bold", pad=12)
    plt.tight_layout()

    if save:
        out = ARTEFACTS_DIR / "shap_bar.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        logger.info("Saved SHAP bar plot → %s", out)

    plt.show()


def plot_waterfall(
    shap_values: shap.Explanation,
    sample_idx: int = 0,
    feature_names: list[str] | None = None,
    max_display: int = 15,
    save: bool = True,
) -> None:
    """
    Waterfall plot for a single prediction — explains *why* this customer was flagged.

    Args:
        shap_values:   Output of compute_shap_values().
        sample_idx:    Index of the sample to explain.
        feature_names: Column name list.
        max_display:   Maximum number of features to show.
        save:          If True, save PNG to models/artefacts/.
    """
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.waterfall(
        shap_values[sample_idx],
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Waterfall — Customer #{sample_idx} Explanation",
              fontweight="bold", pad=12)
    plt.tight_layout()

    if save:
        out = ARTEFACTS_DIR / f"shap_waterfall_{sample_idx}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        logger.info("Saved SHAP waterfall plot → %s", out)

    plt.show()


def plot_dependence(
    shap_values: shap.Explanation,
    feature: str | int,
    interaction_feature: str | int = "auto",
    feature_names: list[str] | None = None,
    save: bool = True,
) -> None:
    """
    Dependence plot — shows how a single feature's SHAP value varies with its value.

    Args:
        shap_values:         Output of compute_shap_values().
        feature:             Feature name (str) or index (int) to plot.
        interaction_feature: Second feature to colour by ('auto' = strongest interaction).
        feature_names:       Column name list.
        save:                If True, save PNG to models/artefacts/.
    """
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.scatter(
        shap_values[:, feature],
        color=shap_values[:, interaction_feature] if interaction_feature != "auto" else shap_values,
        show=False,
    )
    feature_label = feature if isinstance(feature, str) else (
        feature_names[feature] if feature_names else str(feature)
    )
    plt.title(f"SHAP Dependence — {feature_label}", fontweight="bold", pad=12)
    plt.tight_layout()

    if save:
        label = feature_label.replace(" ", "_").replace("/", "_")
        out = ARTEFACTS_DIR / f"shap_dependence_{label}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        logger.info("Saved SHAP dependence plot → %s", out)

    plt.show()


# ── Convenience summary table ─────────────────────────────────────────────────

def top_features(
    shap_values: shap.Explanation,
    feature_names: list[str] | None = None,
    n: int = 10,
) -> list[tuple[str, float]]:
    """
    Return the top-n features by mean absolute SHAP value.

    Args:
        shap_values:   Output of compute_shap_values().
        feature_names: Column name list.
        n:             Number of top features to return.

    Returns:
        List of (feature_name, mean_abs_shap) tuples, descending.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    indices = np.argsort(mean_abs)[::-1][:n]

    if feature_names:
        names = [feature_names[i] if i < len(feature_names) else f"f{i}"
                 for i in indices]
    else:
        names = [f"f{i}" for i in indices]

    return list(zip(names, mean_abs[indices].tolist()))
