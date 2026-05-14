"""
tests/test_tune.py
------------------
Unit tests for src/models/tune.py

Uses synthetic data and a very small n_trials count so tests run fast.
MLflow logging is disabled in all tests.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.models.tune import (
    run_study,
    get_best_model,
    plot_optimization_history,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_data():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 10))
    y = (X[:, 0] + rng.standard_normal(300) * 0.5 > 0).astype(int)
    split = 240
    return X[:split], X[split:], y[:split], y[split:]


@pytest.fixture()
def fast_study(synthetic_data):
    """Run a tiny 5-trial study for testing purposes."""
    X_train, _, y_train, _ = synthetic_data
    return run_study(X_train, y_train, n_trials=5, show_progress=False)


# ── run_study ─────────────────────────────────────────────────────────────────

class TestRunStudy:
    def test_returns_study_object(self, fast_study):
        import optuna
        assert isinstance(fast_study, optuna.Study)

    def test_correct_number_of_trials(self, synthetic_data):
        X_train, _, y_train, _ = synthetic_data
        study = run_study(X_train, y_train, n_trials=3, show_progress=False)
        assert len(study.trials) == 3

    def test_best_value_above_random(self, fast_study):
        assert fast_study.best_value > 0.5

    def test_best_params_contains_n_estimators(self, fast_study):
        assert "n_estimators" in fast_study.best_params

    def test_best_params_contains_learning_rate(self, fast_study):
        assert "learning_rate" in fast_study.best_params

    def test_direction_is_maximize(self, fast_study):
        assert fast_study.direction.name == "MAXIMIZE"


# ── get_best_model ────────────────────────────────────────────────────────────

class TestGetBestModel:
    def test_returns_fitted_model(self, fast_study, synthetic_data):
        from sklearn.utils.validation import check_is_fitted
        X_train, _, y_train, _ = synthetic_data
        model = get_best_model(fast_study, X_train, y_train)
        check_is_fitted(model)

    def test_can_predict_proba(self, fast_study, synthetic_data):
        X_train, X_test, y_train, _ = synthetic_data
        model = get_best_model(fast_study, X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)

    def test_proba_sums_to_one(self, fast_study, synthetic_data):
        X_train, X_test, y_train, _ = synthetic_data
        model = get_best_model(fast_study, X_train, y_train)
        proba = model.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ── plot_optimization_history ─────────────────────────────────────────────────

class TestPlotOptimizationHistory:
    def test_returns_axes(self, fast_study):
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_optimization_history(fast_study)
        import matplotlib.pyplot as plt
        assert ax is not None
        plt.close("all")
