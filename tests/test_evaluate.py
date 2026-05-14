"""
tests/test_evaluate.py
----------------------
Unit tests for src/models/evaluate.py
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LogisticRegression


from src.models.evaluate import compute_metrics, log_confusion_matrix, print_report


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def trained_model():
    """Tiny logistic regression fitted on synthetic binary data."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    y = (X[:, 0] + rng.standard_normal(200) * 0.5 > 0).astype(int)
    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(X, y)
    return model, X, y


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_all_keys(self, trained_model):
        model, X, y = trained_model
        metrics = compute_metrics(model, X, y)
        for key in ("roc_auc", "f1", "accuracy", "precision", "recall"):
            assert key in metrics

    def test_auc_between_0_and_1(self, trained_model):
        model, X, y = trained_model
        metrics = compute_metrics(model, X, y)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_f1_between_0_and_1(self, trained_model):
        model, X, y = trained_model
        metrics = compute_metrics(model, X, y)
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_accuracy_between_0_and_1(self, trained_model):
        model, X, y = trained_model
        metrics = compute_metrics(model, X, y)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_values_are_rounded_to_4dp(self, trained_model):
        model, X, y = trained_model
        metrics = compute_metrics(model, X, y)
        for v in metrics.values():
            assert round(v, 4) == v

    def test_custom_threshold_changes_predictions(self, trained_model):
        model, X, y = trained_model
        metrics_50 = compute_metrics(model, X, y, threshold=0.5)
        metrics_90 = compute_metrics(model, X, y, threshold=0.9)
        # Very high threshold → lower recall
        assert metrics_90["recall"] <= metrics_50["recall"]


# ── log_confusion_matrix ──────────────────────────────────────────────────────

class TestLogConfusionMatrix:
    def test_returns_path(self, trained_model, tmp_path, monkeypatch):
        model, X, y = trained_model
        import src.models.evaluate as ev_module
        monkeypatch.setattr(ev_module, "ARTEFACTS_DIR", tmp_path)
        path = log_confusion_matrix(model, X, y, "TestModel")
        assert isinstance(path, Path)

    def test_file_created(self, trained_model, tmp_path, monkeypatch):
        model, X, y = trained_model
        import src.models.evaluate as ev_module
        monkeypatch.setattr(ev_module, "ARTEFACTS_DIR", tmp_path)
        path = log_confusion_matrix(model, X, y, "TestModel")
        assert path.exists()

    def test_filename_contains_model_name(self, trained_model, tmp_path, monkeypatch):
        model, X, y = trained_model
        import src.models.evaluate as ev_module
        monkeypatch.setattr(ev_module, "ARTEFACTS_DIR", tmp_path)
        path = log_confusion_matrix(model, X, y, "MyClassifier")
        assert "myclassifier" in path.name


# ── print_report ──────────────────────────────────────────────────────────────

class TestPrintReport:
    def test_returns_string(self, trained_model):
        model, X, y = trained_model
        report = print_report(model, X, y, model_name="LR")
        assert isinstance(report, str)

    def test_contains_precision_recall(self, trained_model):
        model, X, y = trained_model
        report = print_report(model, X, y)
        assert "precision" in report
        assert "recall" in report
