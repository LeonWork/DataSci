"""
tests/test_api.py
-----------------
Integration tests for the FastAPI prediction endpoints.

Uses TestClient with a mocked predictor so no real model file is needed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import CustomerFeatures

# ── Sample payloads ───────────────────────────────────────────────────────────

SAMPLE_CUSTOMER = {
    "customerID": "TEST-001",
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0,
}

MOCK_PREDICTION = {
    "customer_id": "TEST-001",
    "churn_probability": 0.78,
    "churn_prediction": True,
    "risk_level": "High",
    "top_factors": [
        {"feature": "tenure", "shap_value": 0.15, "direction": "increases_risk"},
        {"feature": "Contract_Month-to-month", "shap_value": 0.12, "direction": "increases_risk"},
    ],
    "model_version": "1.0.0",
    "timestamp": datetime.now(timezone.utc).isoformat(),
}


@pytest.fixture()
def client():
    """TestClient with predictor mocked as loaded and working."""
    with patch("src.api.main.predictor") as mock_pred:
        mock_pred.is_loaded = True
        mock_pred.model.__class__.__name__ = "XGBClassifier"
        mock_pred.feature_names = ["tenure", "MonthlyCharges"]
        mock_pred.training_metrics = {"roc_auc": 0.84}
        mock_pred.predict.return_value = MOCK_PREDICTION
        yield TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_is_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_model_loaded_true(self, client):
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True

    def test_has_version(self, client):
        resp = client.get("/health")
        assert "version" in resp.json()


# ── /model-info ───────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_returns_200(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_has_model_type(self, client):
        resp = client.get("/model-info")
        assert "model_type" in resp.json()

    def test_has_feature_names(self, client):
        resp = client.get("/model-info")
        assert "feature_names" in resp.json()
        assert isinstance(resp.json()["feature_names"], list)

    def test_has_n_features(self, client):
        resp = client.get("/model-info")
        assert resp.json()["n_features"] == 2


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_200(self, client):
        resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert resp.status_code == 200

    def test_has_churn_probability(self, client):
        resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert "churn_probability" in resp.json()

    def test_probability_in_range(self, client):
        resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        prob = resp.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_has_risk_level(self, client):
        resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert resp.json()["risk_level"] in ("Low", "Medium", "High")

    def test_has_top_factors(self, client):
        resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert isinstance(resp.json()["top_factors"], list)

    def test_factor_has_required_keys(self, client):
        resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        if resp.json()["top_factors"]:
            factor = resp.json()["top_factors"][0]
            assert "feature" in factor
            assert "shap_value" in factor
            assert "direction" in factor

    def test_has_timestamp(self, client):
        resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert "timestamp" in resp.json()

    def test_has_model_version(self, client):
        resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert "model_version" in resp.json()

    def test_invalid_payload_returns_422(self, client):
        resp = client.post("/predict", json={"tenure": "not-a-number"})
        assert resp.status_code == 422

    def test_missing_fields_use_defaults(self, client):
        # Only required field is nothing — all have defaults
        resp = client.post("/predict", json={})
        assert resp.status_code == 200

    def test_customer_id_echoed(self, client):
        resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert resp.json()["customer_id"] == "TEST-001"


# ── /predict-batch ────────────────────────────────────────────────────────────

class TestPredictBatch:
    def test_returns_200(self, client):
        payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
        resp = client.post("/predict-batch", json=payload)
        assert resp.status_code == 200

    def test_total_count_matches(self, client):
        payload = {"customers": [SAMPLE_CUSTOMER]}
        resp = client.post("/predict-batch", json=payload)
        assert resp.json()["total"] == 1

    def test_has_high_risk_count(self, client):
        payload = {"customers": [SAMPLE_CUSTOMER]}
        resp = client.post("/predict-batch", json=payload)
        assert "high_risk_count" in resp.json()

    def test_batch_too_large_returns_400(self, client):
        payload = {"customers": [SAMPLE_CUSTOMER] * 101}
        resp = client.post("/predict-batch", json=payload)
        assert resp.status_code == 400


# ── Unloaded model ────────────────────────────────────────────────────────────

class TestUnloadedModel:
    def test_predict_503_when_model_not_loaded(self):
        with patch("src.api.main.predictor") as mock_pred:
            mock_pred.is_loaded = False
            c = TestClient(app)
            resp = c.post("/predict", json=SAMPLE_CUSTOMER)
            assert resp.status_code == 503
