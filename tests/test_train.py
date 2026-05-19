"""
tests/test_train.py
-------------------
Unit tests for src/models/train.py

MLflow logging is disabled (log_to_mlflow=False) in all tests so
no tracking server or filesystem writes are required.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.train import (
    train_model,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_all_baselines,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_data():
    """200-sample binary dataset — no CSV needed."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))
    y = (X[:, 0] + rng.standard_normal(200) * 0.5 > 0).astype(int)
    split = 160
    return X[:split], X[split:], y[:split], y[split:]


# ── train_model (generic) ─────────────────────────────────────────────────────

class TestTrainModel:
    def test_returns_model_key(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegression(max_iter=200, random_state=0)
        result = train_model(model, X_tr, y_tr, X_te, y_te,
                             "LR", {"max_iter": 200}, log_to_mlflow=False)
        assert "model" in result

    def test_returns_metrics_key(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegression(max_iter=200, random_state=0)
        result = train_model(model, X_tr, y_tr, X_te, y_te,
                             "LR", {}, log_to_mlflow=False)
        assert "metrics" in result

    def test_run_id_is_none_without_mlflow(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegression(max_iter=200, random_state=0)
        result = train_model(model, X_tr, y_tr, X_te, y_te,
                             "LR", {}, log_to_mlflow=False)
        assert result["run_id"] is None

    def test_model_is_fitted(self, synthetic_data):
        from sklearn.utils.validation import check_is_fitted
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegression(max_iter=200, random_state=0)
        result = train_model(model, X_tr, y_tr, X_te, y_te,
                             "LR", {}, log_to_mlflow=False)
        check_is_fitted(result["model"])


# ── train_logistic_regression ─────────────────────────────────────────────────

class TestTrainLogisticRegression:
    def test_auc_above_random(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        result = train_logistic_regression(
            X_tr, y_tr, X_te, y_te, log_to_mlflow=False
        )
        assert result["metrics"]["roc_auc"] > 0.5

    def test_custom_params_applied(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        result = train_logistic_regression(
            X_tr, y_tr, X_te, y_te,
            params={"C": 0.01},
            log_to_mlflow=False,
        )
        assert result["model"].C == 0.01


# ── train_random_forest ───────────────────────────────────────────────────────

class TestTrainRandomForest:
    def test_auc_above_random(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        result = train_random_forest(
            X_tr, y_tr, X_te, y_te,
            params={"n_estimators": 50},   # fast for tests
            log_to_mlflow=False,
        )
        assert result["metrics"]["roc_auc"] > 0.5

    def test_metrics_keys_present(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        result = train_random_forest(
            X_tr, y_tr, X_te, y_te,
            params={"n_estimators": 50},
            log_to_mlflow=False,
        )
        for key in ("roc_auc", "f1", "accuracy", "precision", "recall"):
            assert key in result["metrics"]


# ── train_xgboost ─────────────────────────────────────────────────────────────

class TestTrainXGBoost:
    def test_auc_above_random(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        result = train_xgboost(
            X_tr, y_tr, X_te, y_te,
            params={"n_estimators": 50, "max_depth": 3},
            log_to_mlflow=False,
        )
        assert result["metrics"]["roc_auc"] > 0.5


# ── train_all_baselines ───────────────────────────────────────────────────────

class TestTrainAllBaselines:
    def test_returns_three_models(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        # Use fast params to keep test suite quick
        results = train_all_baselines(
            X_tr, y_tr, X_te, y_te, log_to_mlflow=False
        )
        assert set(results.keys()) == {"LogisticRegression", "RandomForest", "XGBoost", "LightGBM"}

    def test_all_have_metrics(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        results = train_all_baselines(
            X_tr, y_tr, X_te, y_te, log_to_mlflow=False
        )
        for name, r in results.items():
            assert "roc_auc" in r["metrics"], f"Missing roc_auc for {name}"

    def test_xgboost_auc_above_random(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        results = train_all_baselines(
            X_tr, y_tr, X_te, y_te, log_to_mlflow=False
        )
        assert results["XGBoost"]["metrics"]["roc_auc"] > 0.5
