"""
src/api/main.py
---------------
FastAPI application for the Customer Churn Prediction API.

Endpoints
---------
GET  /health          → liveness check
GET  /model-info      → model metadata and feature list
POST /auth/signup     → create local web account
POST /auth/login      → authenticate local web account
POST /predict         → single-customer churn prediction
POST /predict-batch   → batch predictions (up to 100 customers)

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path

import pandas as pd
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.api.auth import (
    AuthenticatedUser,
    authenticate_user,
    company_name,
    create_session_token_for_user,
    create_user,
    signup_enabled,
    signup_requires_invite,
    verify_session_token,
    hash_password,
)
from src.api.model_loader import router
from src.api.schemas import (
    AdminSummaryResponse,
    AuthConfigResponse,
    AuthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CsvPredictionResponse,
    CustomerFeatures,
    FactorImpact,
    HealthResponse,
    LearningStatusResponse,
    LearningUploadResponse,
    LoginRequest,
    ModelInfoResponse,
    PredictionResponse,
    SignupRequest,
    WorkspaceResponse,
    CompanyOnboardRequest,
    CompanyOnboardResponse,
    ImpersonateRequest,
)
from src.api.storage import (
    admin_summary as build_admin_summary,
    init_storage,
    record_learning_rows,
    list_learning_rows,
    learning_status_counts,
    update_learning_row_statuses,
    record_prediction_event,
    record_upload_batch,
    storage_backend as product_storage_backend,
    upsert_workspace_member,
    workspace_overview,
    create_db_user,
    ensure_company,
    record_audit_event,
    get_company_schema,
    latest_training_status,
    clear_company_data,
    save_company_schema,
    start_training_run,
    finish_training_run,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
FEEDBACK_PATH = (
    Path("/tmp") / "churnguard_company_feedback.csv"
    if os.getenv("VERCEL")
    else ROOT / "data" / "company_feedback.csv"
)
BATCH_REQUIRED_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
LEARNING_MAX_ROWS = 5000
SCORING_MAX_ROWS = 500
CHURN_VALUE_MAP = {
    "yes": "Yes",
    "y": "Yes",
    "true": "Yes",
    "1": "Yes",
    "no": "No",
    "n": "No",
    "false": "No",
    "0": "No",
}
CSV_ALLOWED_VALUES = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {"Yes", "No"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}
CSV_NUMERIC_RANGES = {
    "tenure": (0, 72),
    "MonthlyCharges": (0, 999),
    "TotalCharges": (0, None),
}
SCHEMA_METADATA_COLS = {"customerID", "customer_id", "Churn", "source_file", "uploaded_at"}
GLOBAL_COMPANY_IDS = {"default", "global"}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize storage on startup."""
    init_storage()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts the probability of a telecom customer churning "
        "using a Random Forest model trained on the IBM Telco dataset. "
        "Every prediction includes SHAP-based feature explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_response(result: dict) -> PredictionResponse:
    return PredictionResponse(
        customer_id=result["customer_id"],
        churn_probability=result["churn_probability"],
        churn_prediction=result["churn_prediction"],
        risk_level=result["risk_level"],
        top_factors=[FactorImpact(**f) for f in result["top_factors"]],
        model_version=result["model_version"],
        timestamp=result["timestamp"],
    )


def _require_model(company_id: str) -> None:
    try:
        router.get_predictor(company_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded for this workspace. "
                "Run 'python scripts/train_and_save.py' to generate model artefacts."
            ),
        )


def _require_session(authorization: str | None = Header(default=None)) -> AuthenticatedUser:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to use this workspace.",
        )
    session = verify_session_token(token)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Sign in again.",
        )
    upsert_workspace_member(
        company_id=session.company_id,
        username=session.username,
        email=session.email,
        role=session.role,
        company_name=company_name(),
    )
    return session


async def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    try:
        return pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc


def _validate_customer_columns(df: pd.DataFrame) -> None:
    missing = [column for column in BATCH_REQUIRED_COLS if column not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV is missing required columns: {missing}")


def _csv_issue(row: int | None, column: str, code: str, message: str) -> dict:
    return {
        "row": row,
        "column": column,
        "code": code,
        "message": message,
    }


def _raise_csv_validation_error(issues: list[dict]) -> None:
    preview = issues[:50]
    raise HTTPException(
        status_code=400,
        detail={
            "message": "CSV validation failed. Fix the listed issues and upload again.",
            "errors": preview,
            "error_count": len(issues),
            "truncated": len(issues) > len(preview),
        },
    )


def _default_telco_schema() -> dict:
    categorical = {column: sorted(values) for column, values in CSV_ALLOWED_VALUES.items()}
    return {
        "numerical": list(CSV_NUMERIC_RANGES.keys()),
        "categorical": categorical,
    }


def _is_numeric_like(series: pd.Series) -> bool:
    non_empty = series.dropna()
    non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
    if non_empty.empty:
        return False
    converted = pd.to_numeric(non_empty, errors="coerce")
    return float(converted.notna().mean()) >= 0.9


def _infer_schema_from_df(df: pd.DataFrame, company_id: str) -> dict:
    if company_id in GLOBAL_COMPANY_IDS:
        return _default_telco_schema()

    num_cols = []
    cat_cols = {}
    for col in df.columns:
        if col in SCHEMA_METADATA_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or _is_numeric_like(df[col]):
            num_cols.append(col)
        else:
            unique_vals = df[col].dropna().astype(str).unique().tolist()
            cat_cols[col] = unique_vals[:20]
    return {"numerical": num_cols, "categorical": cat_cols}


def _schema_columns(schema: dict) -> tuple[list[str], dict]:
    numerical = list(schema.get("numerical", []))
    categorical = schema.get("categorical", {})
    if isinstance(categorical, list):
        categorical = {column: [] for column in categorical}
    return numerical, categorical


def _normalize_schema_payload(schema: dict) -> dict:
    numerical = schema.get("numerical", [])
    categorical = schema.get("categorical", {})
    if not isinstance(numerical, list):
        raise HTTPException(status_code=400, detail="`numerical` must be a list of column names.")
    if isinstance(categorical, list):
        categorical = {column: [] for column in categorical}
    if not isinstance(categorical, dict):
        raise HTTPException(status_code=400, detail="`categorical` must be an object or list.")

    clean_numerical = []
    for column in numerical:
        value = str(column).strip()
        if value and value not in clean_numerical and value not in SCHEMA_METADATA_COLS:
            clean_numerical.append(value)

    clean_categorical = {}
    for column, options in categorical.items():
        key = str(column).strip()
        if not key or key in SCHEMA_METADATA_COLS or key in clean_numerical:
            continue
        if options is None:
            values = []
        elif isinstance(options, list):
            values = []
            for option in options:
                text = str(option).strip()
                if text and text not in values:
                    values.append(text)
        else:
            raise HTTPException(status_code=400, detail=f"`categorical.{key}` must be a list.")
        clean_categorical[key] = values[:50]

    if not clean_numerical and not clean_categorical:
        raise HTTPException(status_code=400, detail="Schema needs at least one feature column.")
    return {"numerical": clean_numerical, "categorical": clean_categorical}


def _validate_customer_csv(df: pd.DataFrame, *, company_id: str, require_churn: bool, max_rows: int) -> tuple[pd.DataFrame, dict]:
    schema = get_company_schema(company_id)
    if not schema:
        schema = _infer_schema_from_df(df, company_id)
        if company_id not in GLOBAL_COMPANY_IDS:
            save_company_schema(company_id, schema)
            logger.info("Inferred and saved new schema for %s", company_id)

    numerical_cols, categorical_dict = _schema_columns(schema)
    cat_keys = list(categorical_dict.keys())
    required_columns = numerical_cols + cat_keys + (["Churn"] if require_churn else [])
    issues: list[dict] = []
    
    if df.empty:
        issues.append(_csv_issue(None, "file", "empty_csv", "CSV has headers but no customer rows."))
    if len(df) > max_rows:
        issues.append(
            _csv_issue(
                None,
                "file",
                "too_many_rows",
                f"CSV has {len(df)} rows. Limit this upload to {max_rows} rows.",
            )
        )

    missing = [column for column in required_columns if column not in df.columns]
    for column in missing:
        issues.append(
            _csv_issue(
                None,
                column,
                "missing_column",
                f"Add the required `{column}` column.",
            )
        )
        
    if not missing:
        # Row-by-row validation of numerical and categorical constraints
        for idx, row in df.iterrows():
            row_num = idx + 2 # 1-indexed row number (row 1 is headers)
            
            # 1. Numerical validation
            for col in numerical_cols:
                val = row.get(col)
                if pd.isna(val) or val == "":
                    continue
                try:
                    float(val)
                except ValueError:
                    issues.append(
                        _csv_issue(
                            row_num,
                            col,
                            "invalid_number",
                            f"Row {row_num}: `{col}` must be a valid number.",
                        )
                    )
                    
            # 2. Categorical validation
            if isinstance(categorical_dict, dict):
                for col, options in categorical_dict.items():
                    val = row.get(col)
                    if pd.isna(val) or val == "":
                        continue
                    if options and str(val) not in [str(o) for o in options]:
                        issues.append(
                            _csv_issue(
                                row_num,
                                col,
                                "invalid_value",
                                f"Row {row_num}: `{col}` must be one of the allowed options: {options}.",
                            )
                        )
                        
            # 3. Churn validation if required
            if require_churn and "Churn" in df.columns:
                val = row.get("Churn")
                if pd.isna(val) or str(val).strip().lower() not in ["yes", "no", "1", "0", "true", "false"]:
                    issues.append(
                        _csv_issue(
                            row_num,
                            "Churn",
                            "invalid_churn",
                            f"Row {row_num}: `Churn` outcome must be Yes or No.",
                        )
                    )
                        
    if issues:
        _raise_csv_validation_error(issues)

    clean = df.copy()
    for col in numerical_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    if require_churn and "Churn" in clean.columns:
        clean["Churn"] = _normalize_churn_values(clean["Churn"])

    return clean, schema


def _prediction_payload(payload: dict, company_id: str) -> dict:
    schema = get_company_schema(company_id)
    if not schema and company_id in GLOBAL_COMPANY_IDS:
        try:
            return CustomerFeatures.model_validate(payload).model_dump()
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

    raw, _ = _validate_customer_csv(
        pd.DataFrame([payload]),
        company_id=company_id,
        require_churn=False,
        max_rows=1,
    )
    customer = raw.iloc[0].to_dict()
    customer.setdefault("customerID", payload.get("customerID") or payload.get("customer_id") or "WEB-USER")
    return customer


def _normalize_churn_values(values: pd.Series) -> pd.Series:
    normalized = values.map(lambda value: CHURN_VALUE_MAP.get(str(value).strip().lower()))
    invalid = sorted({
        str(value)
        for value, mapped in zip(values, normalized)
        if pd.isna(mapped)
    })
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=(
                "Churn must contain known Yes/No outcomes. "
                f"Invalid values: {invalid}"
            ),
        )
    return normalized


def _read_feedback_store() -> pd.DataFrame:
    if not FEEDBACK_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(FEEDBACK_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read learning queue: {exc}",
        ) from exc


def _learning_storage_backend() -> str:
    csv_backend = "ephemeral /tmp training csv" if os.getenv("VERCEL") else "local training csv"
    return f"{product_storage_backend()} + {csv_backend}"


def _model_auc(company_id: str) -> float | None:
    try:
        p = router.get_predictor(company_id)
        value = p.training_metrics.get("roc_auc") if p.training_metrics else None
        return float(value) if value is not None else None
    except FileNotFoundError:
        return None


def _model_label(predictor) -> str:
    family = getattr(predictor, "model_family", "")
    if isinstance(family, str) and family:
        return family
    return type(predictor.model).__name__


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness check — always returns 200 even if model isn't loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=True,
        version="1.0.0",
    )


@app.get("/auth/config", response_model=AuthConfigResponse, tags=["Authentication"])
def auth_config() -> AuthConfigResponse:
    enabled = signup_enabled()
    invite = signup_requires_invite()
    if not enabled:
        message = "Private workspace. Sign in with your admin credentials."
    elif invite:
        message = "Invite-only workspace. Enter the invite code to create an account."
    else:
        message = "Account creation is open for this workspace."
    return AuthConfigResponse(
        signup_enabled=enabled,
        signup_requires_invite=invite,
        message=message,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info(session: AuthenticatedUser = Depends(_require_session)) -> ModelInfoResponse:
    """Return model metadata, feature count, and training metrics."""
    _require_model(session.company_id)
    p = router.get_predictor(session.company_id)
    return ModelInfoResponse(
        model_type=_model_label(p),
        n_features=len(p.feature_names),
        feature_names=p.feature_names,
        training_metrics={
            **p.training_metrics,
            "model_comparison": getattr(p, "model_comparison", {}) if isinstance(getattr(p, "model_comparison", {}), dict) else {},
            "threshold_report": getattr(p, "threshold_report", []) if isinstance(getattr(p, "threshold_report", []), list) else [],
        },
        version="1.0.0",
    )


@app.get("/admin/summary", response_model=AdminSummaryResponse, tags=["Admin"])
def admin_summary(session: AuthenticatedUser = Depends(_require_session)) -> AdminSummaryResponse:
    """Return workspace-level product metrics for the admin dashboard."""
    _require_model(session.company_id)
    p = router.get_predictor(session.company_id)
    return AdminSummaryResponse(
        **build_admin_summary(
            model_type=_model_label(p),
            model_version="1.0.0",
            model_auc=_model_auc(session.company_id),
            company_id=session.company_id,
        )
    )

@app.get("/admin/schema", tags=["Admin"])
def get_schema(session: AuthenticatedUser = Depends(_require_session)) -> dict:
    """Return the dynamic schema for the company."""
    schema = get_company_schema(session.company_id)
    if not schema:
        # Fallback to empty schema
        return {"numerical": [], "categorical": []}
    return schema


@app.put("/admin/schema", tags=["Admin"])
def update_schema(
    schema: dict = Body(...),
    session: AuthenticatedUser = Depends(_require_session),
) -> dict:
    """Update the active company's schema after admin review."""
    if session.role != "owner":
        raise HTTPException(status_code=403, detail="Only workspace owners can edit schemas.")
    normalized = _normalize_schema_payload(schema)
    save_company_schema(session.company_id, normalized)
    record_audit_event(
        username=session.username,
        event_type="schema_update",
        details=f"Schema updated with {len(normalized['numerical'])} numeric and {len(normalized['categorical'])} categorical columns",
        company_id=session.company_id,
    )
    return {"ok": True, "schema": normalized}


@app.get("/admin/workspace", response_model=WorkspaceResponse, tags=["Admin"])
def workspace(session: AuthenticatedUser = Depends(_require_session)) -> WorkspaceResponse:
    """Return the current workspace and its known members."""
    return WorkspaceResponse(**workspace_overview(session.company_id))


@app.get("/admin/tenants", tags=["Admin"])
def list_tenants(session: AuthenticatedUser = Depends(_require_session)) -> list[dict]:
    """Return all onboarded tenants with stats (Platform Admins only)."""
    if session.role != "owner":
        raise HTTPException(status_code=403, detail="Platform owners only.")
    from src.api.storage import (
        engine as get_engine, companies, app_users, prediction_events,
        learning_rows as lr_table, company_schemas, upload_batches
    )
    from sqlalchemy import select as sel, func
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(
            sel(
                companies.c.id,
                companies.c.name,
                companies.c.created_at,
            )
            .order_by(companies.c.created_at.desc())
        ).fetchall()

        schemas = {r[0] for r in conn.execute(sel(company_schemas.c.company_id)).fetchall()}

        tenant_stats = {}
        for r in rows:
            cid = r[0]
            user_count = conn.execute(
                sel(func.count()).select_from(app_users).where(app_users.c.company_id == cid)
            ).scalar() or 0
            pred_count = conn.execute(
                sel(func.count()).select_from(prediction_events).where(prediction_events.c.company_id == cid)
            ).scalar() or 0
            lr_count = conn.execute(
                sel(func.count()).select_from(lr_table).where(lr_table.c.company_id == cid)
            ).scalar() or 0
            tenant_stats[cid] = {"users": user_count, "preds": pred_count, "lr": lr_count}

    model_dir = Path(__file__).resolve().parents[2] / "models"
    tenants = []
    for r in rows:
        cid, name, created = r[0], r[1], r[2]
        has_model = (model_dir / f"{cid}_model.joblib").exists()
        stats = tenant_stats.get(cid, {})
        tenants.append({
            "company_id": cid,
            "company_name": name,
            "created_at": created.isoformat() if created else None,
            "user_count": stats.get("users", 0),
            "predictions": stats.get("preds", 0),
            "learning_rows": stats.get("lr", 0),
            "has_schema": cid in schemas,
            "has_model": has_model,
        })
    return tenants


@app.post("/admin/retrain", tags=["Admin"])
def retrain_all_models(session: AuthenticatedUser = Depends(_require_session)) -> dict:
    """Trigger the retraining script for all models in a background thread."""
    if session.role != "owner":
        raise HTTPException(status_code=403, detail="Platform owners only.")
    
    import subprocess
    import threading
    import sys
    
    run_id = start_training_run(session.company_id, status="running")

    def run_training(training_run_id: int):
        python_bin = sys.executable
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "train_and_save.py"
        try:
            result = subprocess.run([python_bin, str(script_path), "all"], capture_output=True, text=True)
            if result.returncode == 0:
                finish_training_run(
                    training_run_id,
                    status="succeeded",
                    metrics={"stdout_tail": result.stdout[-2000:]},
                )
            else:
                finish_training_run(
                    training_run_id,
                    status="failed",
                    error_message=(result.stderr or result.stdout)[-2000:],
                )
        except Exception as exc:
            finish_training_run(training_run_id, status="failed", error_message=str(exc))
            
    thread = threading.Thread(target=run_training, args=(run_id,), daemon=True)
    thread.start()
    return {
        "status": "running",
        "run_id": run_id,
        "message": "Retraining triggered successfully in the background.",
    }


@app.get("/admin/retrain/status", tags=["Admin"])
def retrain_status(session: AuthenticatedUser = Depends(_require_session)) -> dict:
    """Return recent model training runs for the active tenant."""
    return {"runs": latest_training_status(session.company_id)}


@app.post("/admin/onboard", response_model=CompanyOnboardResponse, tags=["Admin"])
def onboard_company(
    request: CompanyOnboardRequest,
    session: AuthenticatedUser = Depends(_require_session),
) -> CompanyOnboardResponse:
    """Create a new company and its owner (Platform Admins only)."""
    if session.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only platform owners can onboard new companies."
        )

    ensure_company(request.company_id, request.company_name)

    user = {
        "username": request.owner_username,
        "email": request.owner_email,
        "company_id": request.company_id,
        "company_name": request.company_name,
        "role": "owner",
        "password_hash": hash_password(request.owner_password),
    }

    try:
        create_db_user(user)
        record_audit_event(
            username=user["username"],
            event_type="company_onboard",
            details=f"Provisioned company {request.company_id}",
            company_id=request.company_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return CompanyOnboardResponse(
        ok=True,
        company_id=request.company_id,
        message=f"Company '{request.company_name}' and owner '{request.owner_username}' created successfully.",
    )


@app.post("/admin/impersonate", tags=["Admin"])
def impersonate_company(
    request: ImpersonateRequest,
    session: AuthenticatedUser = Depends(_require_session)
) -> dict:
    """Allow platform owners to switch their session token to another tenant company context."""
    if session.role != "owner":
        raise HTTPException(status_code=403, detail="Platform owners only.")
        
    from src.api.auth import create_session_token_for_user
    impersonated_user = {
        "username": session.username,
        "email": session.email,
        "company_id": request.company_id,
        "role": session.role,  # Keep owner privileges
    }
    new_token = create_session_token_for_user(impersonated_user)
    return {
        "status": "success",
        "access_token": new_token,
        "company_id": request.company_id,
        "role": session.role
    }


@app.post("/admin/delete-tenant", tags=["Admin"])
def delete_tenant(
    request: ImpersonateRequest,
    session: AuthenticatedUser = Depends(_require_session)
) -> dict:
    """Delete a company and all associated schemas, users, predictions, and learning rows."""
    if session.role != "owner":
        raise HTTPException(status_code=403, detail="Platform owners only.")
    if request.company_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default system tenant.")
        
    clear_company_data(request.company_id)
        
    import os
    model_dir = Path(__file__).resolve().parents[2] / "models"
    for suffix in ["_model.joblib", "_pipeline.joblib", "_model_meta.json"]:
        p = model_dir / f"{request.company_id}{suffix}"
        if p.exists():
            try:
                os.remove(p)
            except Exception:
                pass
                
    return {"status": "success", "message": f"Tenant {request.company_id} deleted successfully."}


@app.post("/auth/signup", response_model=AuthResponse, tags=["Authentication"])
def signup(request: SignupRequest) -> AuthResponse:
    ok, message, user = create_user(
        request.username,
        request.email,
        request.password,
        request.confirm_password,
        request.invite_code,
    )
    if not ok or user is None:
        raise HTTPException(status_code=400, detail=message)
    response = AuthResponse(
        ok=True,
        username=user["username"],
        message=message,
        access_token=create_session_token_for_user(user),
        company_id=user["company_id"],
        role=user["role"],
    )
    record_audit_event(
        username=user["username"],
        event_type="signup",
        details="User account created via web signup",
        company_id=user["company_id"]
    )
    return response


@app.post("/auth/login", response_model=AuthResponse, tags=["Authentication"])
def login(request: LoginRequest) -> AuthResponse:
    user = authenticate_user(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    
    record_audit_event(
        username=user["username"],
        event_type="login",
        details="User signed in successfully",
        company_id=user["company_id"]
    )
    
    return AuthResponse(
        ok=True,
        username=user["username"],
        message="Signed in.",
        access_token=create_session_token_for_user(user),
        company_id=user["company_id"],
        role=user["role"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    customer: dict = Body(default_factory=dict),
    session: AuthenticatedUser = Depends(_require_session),
) -> PredictionResponse:
    """
    Predict churn for a single customer.

    Returns the probability, risk level, and top SHAP feature contributions.
    """
    _require_model(session.company_id)
    try:
        p = router.get_predictor(session.company_id)
        payload = _prediction_payload(customer, session.company_id)
        result = p.predict(payload)
        response = _build_response(result)
        record_prediction_event(
            username=session.username,
            customer_id=response.customer_id,
            churn_probability=response.churn_probability,
            risk_level=response.risk_level,
            model_version=response.model_version,
            company_id=session.company_id,
        )
        record_audit_event(
            username=session.username,
            event_type="prediction",
            details=f"Single prediction for {response.customer_id} (Risk: {response.risk_level})",
            company_id=session.company_id,
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
def predict_batch(
    request: BatchPredictionRequest,
    session: AuthenticatedUser = Depends(_require_session),
) -> BatchPredictionResponse:
    """
    Predict churn for a batch of up to 100 customers.
    """
    _require_model(session.company_id)
    if len(request.customers) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 customers per request.",
        )
    predictions = []
    p = router.get_predictor(session.company_id)
    for customer in request.customers:
        try:
            payload = _prediction_payload(customer, session.company_id)
            result = p.predict(payload)
            predictions.append(_build_response(result))
        except Exception as exc:
            logger.error("Error on customer %s: %s", customer.get("customerID", "UNKNOWN"), exc)

    high_risk = sum(1 for p in predictions if p.risk_level == "High")
    for prediction in predictions:
        record_prediction_event(
            username=session.username,
            customer_id=prediction.customer_id,
            churn_probability=prediction.churn_probability,
            risk_level=prediction.risk_level,
            model_version=prediction.model_version,
            company_id=session.company_id,
        )

    record_audit_event(
        username=session.username,
        event_type="predict_batch",
        details=f"Batch scored {len(predictions)} customers ({high_risk} high risk)",
        company_id=session.company_id,
    )

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=high_risk,
    )


@app.post("/predict-csv", response_model=CsvPredictionResponse, tags=["Prediction"])
async def predict_csv(
    file: UploadFile = File(...),
    session: AuthenticatedUser = Depends(_require_session),
) -> CsvPredictionResponse:
    """Score a customer CSV using the current model."""
    _require_model(session.company_id)
    raw = await _read_csv_upload(file)
    raw, schema = _validate_customer_csv(raw, company_id=session.company_id, require_churn=False, max_rows=SCORING_MAX_ROWS)

    rows = []
    p = router.get_predictor(session.company_id)
    for index, record in raw.iterrows():
        customer = record.to_dict()
        customer.setdefault("customerID", f"ROW-{index + 1}")
        result = p.predict(customer)
        top_driver = result["top_factors"][0]["feature"] if result["top_factors"] else ""
        rows.append({
            "customerID": str(customer.get("customerID", f"ROW-{index + 1}")),
            "churn_probability": result["churn_probability"],
            "risk_level": result["risk_level"],
            "top_driver": top_driver,
            "tenure": customer.get("tenure"),
            "contract": customer.get("Contract"),
            "monthly_charges": customer.get("MonthlyCharges"),
        })

    high_risk = sum(1 for row in rows if row["risk_level"] == "High")
    record_upload_batch(
        username=session.username,
        source_file=file.filename,
        upload_type="score",
        row_count=len(rows),
        high_risk_count=high_risk,
        company_id=session.company_id,
    )
    for row in rows:
        record_prediction_event(
            username=session.username,
            customer_id=row["customerID"],
            churn_probability=row["churn_probability"],
            risk_level=row["risk_level"],
            model_version="1.0.0",
            company_id=session.company_id,
        )
        
    record_audit_event(
        username=session.username,
        event_type="predict_csv",
        details=f"CSV upload scored {len(rows)} customers",
        company_id=session.company_id,
    )
    return CsvPredictionResponse(total=len(rows), high_risk_count=high_risk, rows=rows)


@app.post("/learning/upload", response_model=LearningUploadResponse, tags=["Learning"])
async def upload_learning_csv(
    file: UploadFile = File(...),
    session: AuthenticatedUser = Depends(_require_session),
) -> LearningUploadResponse:
    """
    Store labeled company CSV rows for future retraining.

    The file must include the normal customer columns plus a Churn column.
    """
    raw = await _read_csv_upload(file)
    raw, schema = _validate_customer_csv(raw, company_id=session.company_id, require_churn=True, max_rows=LEARNING_MAX_ROWS)

    cat_keys = list(schema.get("categorical", {}).keys()) if isinstance(schema.get("categorical"), dict) else schema.get("categorical", [])
    stored_columns = (["customerID"] if "customerID" in raw.columns else []) + schema.get("numerical", []) + cat_keys + ["Churn"]
    accepted = raw[stored_columns].copy()
    accepted["source_file"] = file.filename
    accepted["uploaded_at"] = pd.Timestamp.utcnow().isoformat()

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_feedback_store()
    stored = pd.concat([existing, accepted], ignore_index=True) if not existing.empty else accepted
    stored.to_csv(FEEDBACK_PATH, index=False)
    batch_id = record_upload_batch(
        username=session.username,
        source_file=file.filename,
        upload_type="learning",
        row_count=len(raw),
        accepted_rows=len(accepted),
        company_id=session.company_id,
    )
    record_learning_rows(
        batch_id=batch_id,
        rows=accepted.to_dict(orient="records"),
        company_id=session.company_id,
    )

    record_audit_event(
        username=session.username,
        event_type="learning_upload",
        details=f"Queued {len(accepted)} labeled rows from {file.filename}",
        company_id=session.company_id,
    )

    return LearningUploadResponse(
        ok=True,
        accepted_rows=len(accepted),
        stored_rows=len(stored),
        message=(
            "Labeled rows saved for retraining. Run the retraining workflow after "
            "reviewing data quality and drift."
        ),
    )


@app.get("/learning/status", response_model=LearningStatusResponse, tags=["Learning"])
def learning_status(_: AuthenticatedUser = Depends(_require_session)) -> LearningStatusResponse:
    """Summarize labeled rows waiting for the next offline retraining run."""
    feedback = _read_feedback_store()
    if feedback.empty:
        return LearningStatusResponse(
            ok=True,
            storage_backend=_learning_storage_backend(),
            storage_path=str(FEEDBACK_PATH),
            stored_rows=0,
            churn_yes_count=0,
            churn_no_count=0,
            latest_uploaded_at=None,
            retraining_command="python scripts/train_and_save.py",
            warning=(
                "Uploads are temporary on Vercel until you connect Blob/Postgres."
                if os.getenv("VERCEL")
                else None
            ),
        )

    churn_counts = feedback.get("Churn", pd.Series(dtype=str)).value_counts()
    latest_uploaded_at = None
    if "uploaded_at" in feedback.columns and not feedback["uploaded_at"].empty:
        latest_uploaded_at = str(feedback["uploaded_at"].max())

    return LearningStatusResponse(
        ok=True,
        storage_backend=_learning_storage_backend(),
        storage_path=str(FEEDBACK_PATH),
        stored_rows=len(feedback),
        churn_yes_count=int(churn_counts.get("Yes", 0)),
        churn_no_count=int(churn_counts.get("No", 0)),
        latest_uploaded_at=latest_uploaded_at,
        retraining_command="python scripts/train_and_save.py",
        warning=(
            "Uploads are temporary on Vercel until you connect Blob/Postgres."
            if os.getenv("VERCEL")
            else None
        ),
    )


@app.get("/learning/review", tags=["Learning"])
def learning_review(session: AuthenticatedUser = Depends(_require_session)) -> dict:
    """Return queued learning rows for owner review before retraining."""
    counts = learning_status_counts(session.company_id)
    rows = list_learning_rows(company_id=session.company_id, status="queued", limit=100)
    return {"counts": counts, "rows": rows}


@app.post("/learning/review", tags=["Learning"])
def review_learning_rows(
    request: dict = Body(...),
    session: AuthenticatedUser = Depends(_require_session),
) -> dict:
    """Approve or reject queued learning rows."""
    if session.role != "owner":
        raise HTTPException(status_code=403, detail="Only workspace owners can review learning rows.")
    status_value = str(request.get("status", "")).strip()
    allowed = {"approved_for_training", "rejected"}
    if status_value not in allowed:
        raise HTTPException(status_code=400, detail=f"Status must be one of {sorted(allowed)}.")
    row_ids = request.get("row_ids", [])
    if not isinstance(row_ids, list):
        raise HTTPException(status_code=400, detail="`row_ids` must be a list.")
    updated = update_learning_row_statuses(
        company_id=session.company_id,
        row_ids=row_ids,
        status=status_value,
    )
    record_audit_event(
        username=session.username,
        event_type="learning_review",
        details=f"{status_value}: {updated} learning rows",
        company_id=session.company_id,
    )
    return {
        "ok": True,
        "updated_rows": updated,
        "counts": learning_status_counts(session.company_id),
    }


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
