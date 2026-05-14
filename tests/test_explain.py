"""
tests/test_explain.py
---------------------
Unit tests for src/models/explain.py

Uses a tiny synthetic dataset and a pre-fitted LR/XGB model so no CSV is needed.
All plots are saved to a tmp_path to avoid cluttering models/artefacts/.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.explain import (
    compute_shap_values,
    top_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_binary():
    rng = np.random.default_rng(99)
    X = rng.standard_normal((120, 8))
    y = (X[:, 0] + rng.standard_normal(120) * 0.3 > 0).astype(int)
    return X, y


@pytest.fixture()
def fitted_xgb(synthetic_binary):
    X, y = synthetic_binary
    model = XGBClassifier(
        n_estimators=30,
        max_depth=3,
        random_state=0,
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model, X, y


# ── compute_shap_values ───────────────────────────────────────────────────────

class TestComputeShapValues:
    def test_returns_explanation(self, fitted_xgb):
        import shap
        model, X, _ = fitted_xgb
        sv = compute_shap_values(model, X)
        assert isinstance(sv, shap.Explanation)

    def test_shape_matches_input(self, fitted_xgb):
        model, X, _ = fitted_xgb
        sv = compute_shap_values(model, X)
        assert sv.values.shape[0] == len(X)

    def test_num_features_matches_columns(self, fitted_xgb):
        model, X, _ = fitted_xgb
        sv = compute_shap_values(model, X)
        assert sv.values.shape[1] == X.shape[1]

    def test_max_samples_respected(self, fitted_xgb):
        model, X, _ = fitted_xgb
        sv = compute_shap_values(model, X, max_samples=20)
        # SHAP values are still computed for all input rows
        assert sv.values.shape[0] == len(X)

    def test_values_are_finite(self, fitted_xgb):
        model, X, _ = fitted_xgb
        sv = compute_shap_values(model, X)
        assert np.isfinite(sv.values).all()


# ── top_features ──────────────────────────────────────────────────────────────

class TestTopFeatures:
    def test_returns_correct_length(self, fitted_xgb):
        model, X, _ = fitted_xgb
        sv = compute_shap_values(model, X)
        result = top_features(sv, n=3)
        assert len(result) == 3

    def test_returns_tuples(self, fitted_xgb):
        model, X, _ = fitted_xgb
        sv = compute_shap_values(model, X)
        result = top_features(sv)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_scores_are_descending(self, fitted_xgb):
        model, X, _ = fitted_xgb
        sv = compute_shap_values(model, X)
        result = top_features(sv)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_feature_names_applied(self, fitted_xgb):
        model, X, _ = fitted_xgb
        sv = compute_shap_values(model, X)
        names = [f"col_{i}" for i in range(X.shape[1])]
        result = top_features(sv, feature_names=names, n=5)
        for name, _ in result:
            assert name.startswith("col_")

    def test_n_capped_at_num_features(self, fitted_xgb):
        model, X, _ = fitted_xgb
        sv = compute_shap_values(model, X)
        result = top_features(sv, n=999)
        assert len(result) == X.shape[1]
