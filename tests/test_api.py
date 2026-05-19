"""
tests/test_api.py
-----------------
Integration tests for the FastAPI prediction endpoints.

Uses TestClient with a mocked predictor so no real model file is needed.
"""

from __future__ import annotations

import os
import tempfile
# Force isolated test database before any other imports trigger engine initialization
os.environ["CHURNGUARD_DB_PATH"] = os.path.join(tempfile.gettempdir(), "churnguard-test-isolated.sqlite3")

from io import BytesIO
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.auth import (
    authenticate_user,
    create_session_token,
    create_session_token_for_user,
    create_user,
    signup_enabled,
)
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
def client(tmp_path, monkeypatch):
    """TestClient with predictor mocked as loaded and working."""
    monkeypatch.setenv("CHURNGUARD_DB_PATH", str(tmp_path / "churnguard-test.sqlite3"))
    
    # Reset engine and clean up DB tables for test isolation
    from src.api.storage import engine, prediction_events, learning_rows, upload_batches, app_users, workspace_members
    eng = engine()
    try:
        with eng.begin() as conn:
            conn.execute(prediction_events.delete())
            conn.execute(learning_rows.delete())
            conn.execute(upload_batches.delete())
            conn.execute(app_users.delete())
            conn.execute(workspace_members.delete())
    except Exception:
        pass
        
    mock_pred = MagicMock()
    mock_pred.is_loaded = True
    mock_pred.model.__class__.__name__ = "RandomForestClassifier"
    mock_pred.feature_names = ["tenure", "MonthlyCharges"]
    mock_pred.training_metrics = {"roc_auc": 0.7897}
    mock_pred.predict.return_value = MOCK_PREDICTION
    
    with patch("src.api.model_loader.router.get_predictor", return_value=mock_pred):
        with patch("src.api.main._require_model"):
            test_client = TestClient(app)
            test_client.headers.update({"Authorization": f"Bearer {create_session_token('admin')}"})
            yield test_client


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

    def test_requires_signed_in_session(self):
        c = TestClient(app)
        resp = c.post("/predict", json=SAMPLE_CUSTOMER)
        assert resp.status_code == 401


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
        from fastapi import HTTPException
        with patch("src.api.main._require_model", side_effect=HTTPException(status_code=503, detail="Unloaded")):
            c = TestClient(app)
            c.headers.update({"Authorization": f"Bearer {create_session_token('admin')}"})
            resp = c.post("/predict", json=SAMPLE_CUSTOMER)
            assert resp.status_code == 503


# ── Learning queue ────────────────────────────────────────────────────────────

def _learning_csv(churn: str = "Yes") -> bytes:
    row = {
        **SAMPLE_CUSTOMER,
        "Churn": churn,
    }
    headers = list(row.keys())
    line = ",".join(str(row[key]) for key in headers)
    return (",".join(headers) + "\n" + line + "\n").encode()


def _scoring_csv() -> bytes:
    headers = list(SAMPLE_CUSTOMER.keys())
    line = ",".join(str(SAMPLE_CUSTOMER[key]) for key in headers)
    return (",".join(headers) + "\n" + line + "\n").encode()


def _csv_from_row(row: dict) -> bytes:
    headers = list(row.keys())
    line = ",".join(str(row[key]) for key in headers)
    return (",".join(headers) + "\n" + line + "\n").encode()


class TestAdminSummary:
    def test_summary_starts_with_zero_counts(self, client):
        resp = client.get("/admin/summary")
        body = resp.json()
        assert resp.status_code == 200
        assert body["total_predictions"] == 0
        assert body["high_risk_predictions"] == 0
        assert body["learning_rows_queued"] == 0
        assert body["model_type"] == "RandomForestClassifier"
        assert body["model_auc"] == 0.7897

    def test_prediction_updates_summary(self, client):
        client.post("/predict", json=SAMPLE_CUSTOMER)
        body = client.get("/admin/summary").json()
        assert body["total_predictions"] == 1
        assert body["high_risk_predictions"] == 1

    def test_csv_upload_updates_batches_and_predictions(self, client):
        resp = client.post(
            "/predict-csv",
            files={"file": ("customers.csv", BytesIO(_scoring_csv()), "text/csv")},
        )
        assert resp.status_code == 200

        body = client.get("/admin/summary").json()
        assert body["total_predictions"] == 1
        assert body["csv_upload_batches"] == 1
        assert body["latest_upload_at"] is not None

    def test_csv_upload_reports_missing_columns(self, client):
        row = {key: value for key, value in SAMPLE_CUSTOMER.items() if key != "Contract"}
        resp = client.post(
            "/predict-csv",
            files={"file": ("bad.csv", BytesIO(_csv_from_row(row)), "text/csv")},
        )
        body = resp.json()
        assert resp.status_code == 400
        assert body["detail"]["message"] == "CSV validation failed. Fix the listed issues and upload again."
        assert body["detail"]["errors"][0]["code"] == "missing_column"
        assert body["detail"]["errors"][0]["column"] == "Contract"

    def test_csv_upload_reports_invalid_values_with_row_numbers(self, client):
        row = {
            **SAMPLE_CUSTOMER,
            "Contract": "Weekly",
            "tenure": 120,
            "MonthlyCharges": "expensive",
        }
        resp = client.post(
            "/predict-csv",
            files={"file": ("bad.csv", BytesIO(_csv_from_row(row)), "text/csv")},
        )
        errors = resp.json()["detail"]["errors"]
        assert resp.status_code == 400
        assert {error["code"] for error in errors} >= {"invalid_value", "invalid_number"}
        assert {error["row"] for error in errors} == {2}

    def test_workspace_metadata_is_available(self, client):
        resp = client.get("/admin/workspace")
        body = resp.json()
        assert resp.status_code == 200
        assert body["company_id"] == "default"
        assert body["company_name"] == "ChurnGuard Pilot"
        assert body["members"][0]["username"] == "admin"
        assert body["members"][0]["role"] == "owner"

    def test_prediction_counts_are_tenant_isolated(self, client):
        company_a = create_session_token_for_user({
            "username": "owner-a",
            "email": "a@example.com",
            "company_id": "company-a",
            "role": "owner",
        })
        company_b = create_session_token_for_user({
            "username": "owner-b",
            "email": "b@example.com",
            "company_id": "company-b",
            "role": "owner",
        })

        client.headers.update({"Authorization": f"Bearer {company_a}"})
        assert client.post("/predict", json=SAMPLE_CUSTOMER).status_code == 200
        assert client.get("/admin/summary").json()["total_predictions"] == 1

        client.headers.update({"Authorization": f"Bearer {company_b}"})
        body = client.get("/admin/summary").json()
        assert body["total_predictions"] == 0
        assert body["high_risk_predictions"] == 0


class TestLearningQueue:
    def test_status_empty_queue(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.main.FEEDBACK_PATH", tmp_path / "feedback.csv")
        resp = client.get("/learning/status")
        body = resp.json()
        assert resp.status_code == 200
        assert body["stored_rows"] == 0
        assert body["retraining_command"] == "python scripts/train_and_save.py"

    def test_upload_stores_labeled_rows(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.main.FEEDBACK_PATH", tmp_path / "feedback.csv")
        resp = client.post(
            "/learning/upload",
            files={"file": ("learning.csv", BytesIO(_learning_csv("Yes")), "text/csv")},
        )
        body = resp.json()
        assert resp.status_code == 200
        assert body["accepted_rows"] == 1
        assert body["stored_rows"] == 1

        status = client.get("/learning/status").json()
        assert status["stored_rows"] == 1
        assert status["churn_yes_count"] == 1
        summary = client.get("/admin/summary").json()
        assert summary["learning_rows_queued"] == 1
        assert summary["csv_upload_batches"] == 1

    def test_upload_rejects_unknown_churn_label(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.main.FEEDBACK_PATH", tmp_path / "feedback.csv")
        resp = client.post(
            "/learning/upload",
            files={"file": ("learning.csv", BytesIO(_learning_csv("Maybe")), "text/csv")},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["errors"][0]["column"] == "Churn"
        assert detail["errors"][0]["code"] == "invalid_churn"


class TestAuthDefaults:
    def test_default_admin_is_disabled_without_explicit_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.auth.USER_STORE_PATH", tmp_path / "users.json")
        monkeypatch.delenv("CHURNGUARD_USERNAME", raising=False)
        monkeypatch.delenv("CHURNGUARD_PASSWORD", raising=False)
        monkeypatch.delenv("CHURNGUARD_PASSWORD_HASH", raising=False)
        monkeypatch.delenv("CHURNGUARD_ENABLE_DEFAULT_ADMIN", raising=False)
        assert authenticate_user("admin", "admin123") is None

    def test_signup_is_disabled_without_env_or_invite(self, monkeypatch):
        monkeypatch.delenv("CHURNGUARD_ENABLE_SIGNUP", raising=False)
        monkeypatch.delenv("CHURNGUARD_SIGNUP_CODE", raising=False)
        assert signup_enabled() is False

    def test_signup_creates_database_user(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHURNGUARD_DB_PATH", str(tmp_path / "auth.sqlite3"))
        monkeypatch.setenv("CHURNGUARD_ENABLE_SIGNUP", "true")
        monkeypatch.delenv("CHURNGUARD_USE_LEGACY_JSON_USERS", raising=False)

        ok, message, user = create_user(
            "analyst1",
            "analyst1@example.com",
            "secure-password",
            "secure-password",
        )

        assert ok is True
        assert message == "Account created."
        assert user["role"] == "analyst"

        authed = authenticate_user("analyst1", "secure-password")
        assert authed is not None
        assert authed["username"] == "analyst1"
        assert authed["company_id"] == "default"
