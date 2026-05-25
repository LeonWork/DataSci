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
    from src.api.storage import engine, prediction_events, learning_rows, upload_batches, app_users, workspace_members, model_training_runs
    eng = engine()
    try:
        with eng.begin() as conn:
            conn.execute(prediction_events.delete())
            conn.execute(learning_rows.delete())
            conn.execute(upload_batches.delete())
            conn.execute(model_training_runs.delete())
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

    def test_dynamic_tenant_payload_accepts_schema_fields(self, client):
        from src.api.storage import save_company_schema

        save_company_schema("acme-dynamic", {
            "numerical": ["days_since_last_purchase", "total_spent"],
            "categorical": {"plan": ["Free", "Pro"]},
        })
        token = create_session_token_for_user({
            "username": "owner-acme",
            "email": "owner@acme.test",
            "company_id": "acme-dynamic",
            "role": "owner",
        })
        client.headers.update({"Authorization": f"Bearer {token}"})

        resp = client.post("/predict", json={
            "customerID": "DYN-001",
            "days_since_last_purchase": "12",
            "total_spent": 830.50,
            "plan": "Pro",
        })

        assert resp.status_code == 200
        assert resp.json()["risk_level"] == "High"

    def test_dynamic_tenant_payload_reports_validation_errors(self, client):
        from src.api.storage import save_company_schema

        save_company_schema("acme-dynamic", {
            "numerical": ["days_since_last_purchase"],
            "categorical": {"plan": ["Free", "Pro"]},
        })
        token = create_session_token_for_user({
            "username": "owner-acme",
            "email": "owner@acme.test",
            "company_id": "acme-dynamic",
            "role": "owner",
        })
        client.headers.update({"Authorization": f"Bearer {token}"})

        resp = client.post("/predict", json={
            "days_since_last_purchase": "soon",
            "plan": "Enterprise",
        })

        assert resp.status_code == 400
        errors = resp.json()["detail"]["errors"]
        assert {error["code"] for error in errors} == {"invalid_number", "invalid_value"}


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

    def test_schema_inference_detects_numeric_strings(self, client):
        row = {
            "customerID": "DYN-CSV-001",
            "days_since_last_purchase": "14",
            "total_spent": "420.50",
            "plan": "Pro",
        }
        token = create_session_token_for_user({
            "username": "owner-dyn",
            "email": "dyn@example.com",
            "company_id": "dynamic-csv",
            "role": "owner",
        })
        client.headers.update({"Authorization": f"Bearer {token}"})

        resp = client.post(
            "/predict-csv",
            files={"file": ("dynamic.csv", BytesIO(_csv_from_row(row)), "text/csv")},
        )

        assert resp.status_code == 200
        from src.api.storage import get_company_schema

        schema = get_company_schema("dynamic-csv")
        assert "days_since_last_purchase" in schema["numerical"]
        assert "total_spent" in schema["numerical"]
        assert "plan" in schema["categorical"]

    def test_retraining_status_records_runs(self, client):
        from src.api.storage import start_training_run, finish_training_run

        run_id = start_training_run("default", status="running")
        finish_training_run(
            run_id,
            status="succeeded",
            model_family="XGBoost_Tuned",
            metrics={"roc_auc": 0.86, "f1": 0.6},
        )

        resp = client.get("/admin/retrain/status")

        assert resp.status_code == 200
        run = resp.json()["runs"][0]
        assert run["status"] == "succeeded"
        assert run["model_family"] == "XGBoost_Tuned"
        assert run["metrics"]["roc_auc"] == 0.86

    def test_owner_can_update_schema(self, client):
        resp = client.put("/admin/schema", json={
            "numerical": ["days_since_last_purchase", "total_spent"],
            "categorical": {
                "plan": ["Free", "Pro"],
                "region": ["West", "East"],
            },
        })

        assert resp.status_code == 200
        schema = resp.json()["schema"]
        assert schema["numerical"] == ["days_since_last_purchase", "total_spent"]
        assert schema["categorical"]["plan"] == ["Free", "Pro"]

    def test_non_owner_cannot_update_schema(self, client):
        token = create_session_token_for_user({
            "username": "viewer",
            "email": "viewer@example.com",
            "company_id": "default",
            "role": "viewer",
        })
        client.headers.update({"Authorization": f"Bearer {token}"})

        resp = client.put("/admin/schema", json={"numerical": ["x"], "categorical": {}})

        assert resp.status_code == 403

    def test_failed_retraining_status_preserves_error(self, client):
        from src.api.storage import start_training_run, finish_training_run

        run_id = start_training_run("default", status="running")
        finish_training_run(run_id, status="failed", error_message="candidate model failed")

        resp = client.get("/admin/retrain/status")

        assert resp.status_code == 200
        run = resp.json()["runs"][0]
        assert run["status"] == "failed"
        assert run["error_message"] == "candidate model failed"


class TestModelPromotion:
    def _write_artifacts(self, model_dir, stem, meta):
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"{stem}_model.joblib").write_text(f"{stem}-model")
        (model_dir / f"{stem}_pipeline.joblib").write_text(f"{stem}-pipeline")
        (model_dir / f"{stem}_model_meta.json").write_text(json.dumps(meta))

    def test_candidate_status_compares_production_and_candidate(self, client, tmp_path, monkeypatch):
        from src.api.storage import start_training_run, finish_training_run

        monkeypatch.setattr("src.api.main.MODELS_DIR", tmp_path)
        self._write_artifacts(tmp_path, "default", {
            "model_family": "RandomForest",
            "metrics": {"roc_auc": 0.80, "f1": 0.58},
        })
        self._write_artifacts(tmp_path, "default_candidate", {
            "artifact_stage": "candidate",
            "model_family": "LightGBM_Tuned",
            "metrics": {"roc_auc": 0.86, "f1": 0.63},
        })
        run_id = start_training_run("default", status="running")
        finish_training_run(
            run_id,
            status="candidate_ready",
            model_family="LightGBM_Tuned",
            metrics={"roc_auc": 0.86, "f1": 0.63},
            artifact_paths={"metadata": str(tmp_path / "default_candidate_model_meta.json")},
        )

        resp = client.get("/admin/model/candidate")

        assert resp.status_code == 200
        body = resp.json()
        assert body["production"]["metadata"]["model_family"] == "RandomForest"
        assert body["candidate"]["metadata"]["model_family"] == "LightGBM_Tuned"
        assert body["candidate"]["can_promote"] is True
        assert body["candidate"]["quality_gate"]["passed"] is True

    def test_promote_candidate_replaces_production_artifacts(self, client, tmp_path, monkeypatch):
        from src.api.storage import start_training_run, finish_training_run, latest_training_status

        monkeypatch.setattr("src.api.main.MODELS_DIR", tmp_path)
        self._write_artifacts(tmp_path, "default", {
            "model_family": "RandomForest",
            "metrics": {"roc_auc": 0.80},
        })
        self._write_artifacts(tmp_path, "default_candidate", {
            "artifact_stage": "candidate",
            "model_family": "LightGBM_Tuned",
            "metrics": {"roc_auc": 0.86},
        })
        run_id = start_training_run("default", status="running")
        finish_training_run(
            run_id,
            status="candidate_ready",
            model_family="LightGBM_Tuned",
            metrics={"roc_auc": 0.86},
        )

        resp = client.post("/admin/model/promote", json={})

        assert resp.status_code == 200
        assert json.loads((tmp_path / "default_model_meta.json").read_text())["model_family"] == "LightGBM_Tuned"
        assert (tmp_path / "default_model.joblib").read_text() == "default_candidate-model"
        statuses = latest_training_status("default")
        assert statuses[0]["status"] == "promoted"

    def test_worse_candidate_is_blocked_by_quality_gate(self, client, tmp_path, monkeypatch):
        from src.api.storage import start_training_run, finish_training_run, latest_training_status

        monkeypatch.setattr("src.api.main.MODELS_DIR", tmp_path)
        self._write_artifacts(tmp_path, "default", {
            "model_family": "RandomForest",
            "metrics": {"roc_auc": 0.86, "pr_auc": 0.62, "f1": 0.61, "brier": 0.14},
        })
        self._write_artifacts(tmp_path, "default_candidate", {
            "artifact_stage": "candidate",
            "model_family": "WeakCandidate",
            "metrics": {"roc_auc": 0.78, "pr_auc": 0.50, "f1": 0.52, "brier": 0.19},
        })
        run_id = start_training_run("default", status="running")
        finish_training_run(run_id, status="candidate_ready", model_family="WeakCandidate")

        resp = client.post("/admin/model/promote", json={})

        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["quality_gate"]["passed"] is False
        assert detail["quality_gate"]["blockers"]
        assert json.loads((tmp_path / "default_model_meta.json").read_text())["model_family"] == "RandomForest"
        assert latest_training_status("default")[0]["status"] == "candidate_ready"

    def test_owner_can_force_promote_blocked_candidate(self, client, tmp_path, monkeypatch):
        from src.api.storage import start_training_run, finish_training_run, latest_training_status

        monkeypatch.setattr("src.api.main.MODELS_DIR", tmp_path)
        self._write_artifacts(tmp_path, "default", {
            "model_family": "RandomForest",
            "metrics": {"roc_auc": 0.86, "pr_auc": 0.62, "f1": 0.61, "brier": 0.14},
        })
        self._write_artifacts(tmp_path, "default_candidate", {
            "artifact_stage": "candidate",
            "model_family": "BusinessOverrideCandidate",
            "metrics": {"roc_auc": 0.78, "pr_auc": 0.50, "f1": 0.52, "brier": 0.19},
        })
        run_id = start_training_run("default", status="running")
        finish_training_run(run_id, status="candidate_ready", model_family="BusinessOverrideCandidate")

        resp = client.post("/admin/model/promote", json={"force": True})

        assert resp.status_code == 200
        assert resp.json()["forced"] is True
        assert resp.json()["quality_gate"]["passed"] is False
        assert json.loads((tmp_path / "default_model_meta.json").read_text())["model_family"] == "BusinessOverrideCandidate"
        latest_run = latest_training_status("default")[0]
        assert latest_run["status"] == "promoted"
        assert latest_run["metrics"]["force_promoted"] is True

    def test_promote_candidate_marks_included_learning_rows_used(self, client, tmp_path, monkeypatch):
        from src.api.storage import (
            finish_training_run,
            latest_training_status,
            list_learning_rows,
            record_learning_rows,
            record_upload_batch,
            start_training_run,
            update_learning_row_statuses,
        )

        monkeypatch.setattr("src.api.main.MODELS_DIR", tmp_path)
        batch_id = record_upload_batch(
            username="admin",
            source_file="learning.csv",
            upload_type="learning",
            row_count=2,
            accepted_rows=2,
            company_id="default",
        )
        record_learning_rows(
            batch_id=batch_id,
            company_id="default",
            rows=[
                {**SAMPLE_CUSTOMER, "customerID": "USED-001", "Churn": "Yes"},
                {**SAMPLE_CUSTOMER, "customerID": "FUTURE-001", "Churn": "No"},
            ],
        )
        row_ids = [row["id"] for row in list_learning_rows(company_id="default", status="queued")]
        update_learning_row_statuses(
            company_id="default",
            row_ids=row_ids,
            status="approved_for_training",
        )
        included_row_id = row_ids[0]

        self._write_artifacts(tmp_path, "default", {
            "model_family": "RandomForest",
            "metrics": {"roc_auc": 0.80},
        })
        self._write_artifacts(tmp_path, "default_candidate", {
            "artifact_stage": "candidate",
            "model_family": "LightGBM_Tuned",
            "metrics": {"roc_auc": 0.86},
            "learning_row_ids": [included_row_id],
        })
        run_id = start_training_run("default", status="running")
        finish_training_run(run_id, status="candidate_ready", model_family="LightGBM_Tuned")

        resp = client.post("/admin/model/promote", json={})

        assert resp.status_code == 200
        assert resp.json()["used_learning_rows"] == 1
        used_rows = list_learning_rows(company_id="default", status="used_in_model")
        approved_rows = list_learning_rows(company_id="default", status="approved_for_training")
        assert [row["id"] for row in used_rows] == [included_row_id]
        assert used_rows[0]["model_training_run_id"] == latest_training_status("default")[0]["id"]
        assert len(approved_rows) == 1

    def test_reject_candidate_leaves_production_artifacts_untouched(self, client, tmp_path, monkeypatch):
        from src.api.storage import start_training_run, finish_training_run, latest_training_status

        monkeypatch.setattr("src.api.main.MODELS_DIR", tmp_path)
        self._write_artifacts(tmp_path, "default", {
            "model_family": "RandomForest",
            "metrics": {"roc_auc": 0.80},
        })
        self._write_artifacts(tmp_path, "default_candidate", {
            "artifact_stage": "candidate",
            "model_family": "LightGBM_Tuned",
            "metrics": {"roc_auc": 0.86},
        })
        run_id = start_training_run("default", status="running")
        finish_training_run(run_id, status="candidate_ready", model_family="LightGBM_Tuned")

        resp = client.post("/admin/model/reject", json={})

        assert resp.status_code == 200
        assert json.loads((tmp_path / "default_model_meta.json").read_text())["model_family"] == "RandomForest"
        assert not (tmp_path / "default_candidate_model.joblib").exists()
        statuses = latest_training_status("default")
        assert statuses[0]["status"] == "rejected"

    def test_non_owner_cannot_promote_candidate(self, client):
        token = create_session_token_for_user({
            "username": "viewer",
            "email": "viewer@example.com",
            "company_id": "default",
            "role": "viewer",
        })
        client.headers.update({"Authorization": f"Bearer {token}"})

        resp = client.post("/admin/model/promote", json={})

        assert resp.status_code == 403


class TestDriftMonitoring:
    def _write_profile_meta(self, model_dir, profile):
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "default_model_meta.json").write_text(json.dumps({
            "model_family": "RandomForest",
            "metrics": {"roc_auc": 0.84},
            "training_profile": profile,
        }))

    def test_drift_monitor_warms_up_before_enough_predictions(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.main.MODELS_DIR", tmp_path)
        self._write_profile_meta(tmp_path, {
            "numeric": {"tenure": {"mean": 10, "std": 2}},
            "categorical": {"Contract": {"top_values": {"Month-to-month": 0.5, "Two year": 0.5}}},
        })

        resp = client.get("/admin/drift")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "warming_up"
        assert body["sample_size"] == 0

    def test_drift_monitor_detects_shifted_recent_inputs(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.main.MODELS_DIR", tmp_path)
        self._write_profile_meta(tmp_path, {
            "numeric": {
                "tenure": {"mean": 5, "std": 1},
                "MonthlyCharges": {"mean": 50, "std": 10},
            },
            "categorical": {
                "Contract": {"top_values": {"Two year": 0.9, "Month-to-month": 0.1}},
            },
        })
        shifted = {
            **SAMPLE_CUSTOMER,
            "tenure": 30,
            "MonthlyCharges": 120,
            "Contract": "Month-to-month",
        }
        for index in range(10):
            payload = {**shifted, "customerID": f"DRIFT-{index}"}
            assert client.post("/predict", json=payload).status_code == 200

        resp = client.get("/admin/drift")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "high"
        assert body["retrain_recommended"] is True
        assert body["sample_size"] == 10
        assert body["features"][0]["score"] >= 0.75

    def test_drift_monitor_reports_missing_training_profile(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.main.MODELS_DIR", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "default_model_meta.json").write_text(json.dumps({
            "model_family": "RandomForest",
            "metrics": {"roc_auc": 0.84},
        }))

        resp = client.get("/admin/drift")

        assert resp.status_code == 200
        assert resp.json()["status"] == "unavailable"


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

    def test_learning_review_approves_and_rejects_rows(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.main.FEEDBACK_PATH", tmp_path / "feedback.csv")
        for churn in ("Yes", "No"):
            resp = client.post(
                "/learning/upload",
                files={"file": (f"learning-{churn}.csv", BytesIO(_learning_csv(churn)), "text/csv")},
            )
            assert resp.status_code == 200

        review = client.get("/learning/review").json()
        row_ids = [row["id"] for row in review["rows"]]
        assert review["counts"]["queued"] == 2

        approve = client.post(
            "/learning/review",
            json={"row_ids": [row_ids[0]], "status": "approved_for_training"},
        )
        reject = client.post(
            "/learning/review",
            json={"row_ids": [row_ids[1]], "status": "rejected"},
        )

        assert approve.status_code == 200
        assert reject.status_code == 200
        counts = client.get("/learning/review").json()["counts"]
        assert counts["approved_for_training"] == 1
        assert counts["rejected"] == 1
        assert counts.get("queued", 0) == 0

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
