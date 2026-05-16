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
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.auth import authenticate_user, create_user
from src.api.model_loader import predictor
from src.api.schemas import (
    AuthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CsvPredictionResponse,
    CustomerFeatures,
    FactorImpact,
    HealthResponse,
    LearningUploadResponse,
    LoginRequest,
    ModelInfoResponse,
    PredictionResponse,
    SignupRequest,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
FEEDBACK_PATH = ROOT / "data" / "company_feedback.csv"
BATCH_REQUIRED_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; nothing to clean up on shutdown."""
    try:
        predictor.load()
        logger.info("API ready — model loaded successfully")
    except FileNotFoundError as exc:
        logger.error("Model artefacts missing: %s", exc)
        logger.error("Run: python scripts/train_and_save.py  to generate them.")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts the probability of a telecom customer churning "
        "using a tuned XGBoost model trained on the IBM Telco dataset. "
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


def _require_model() -> None:
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded. "
                "Run 'python scripts/train_and_save.py' to generate model artefacts."
            ),
        )


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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness check — always returns 200 even if model isn't loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info() -> ModelInfoResponse:
    """Return model metadata, feature count, and training metrics."""
    _require_model()
    return ModelInfoResponse(
        model_type=type(predictor.model).__name__,
        n_features=len(predictor.feature_names),
        feature_names=predictor.feature_names,
        training_metrics=predictor.training_metrics,
        version="1.0.0",
    )


@app.post("/auth/signup", response_model=AuthResponse, tags=["Authentication"])
def signup(request: SignupRequest) -> AuthResponse:
    ok, message, user = create_user(
        request.username,
        request.email,
        request.password,
        request.confirm_password,
    )
    if not ok or user is None:
        raise HTTPException(status_code=400, detail=message)
    return AuthResponse(ok=True, username=user["username"], message=message)


@app.post("/auth/login", response_model=AuthResponse, tags=["Authentication"])
def login(request: LoginRequest) -> AuthResponse:
    user = authenticate_user(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return AuthResponse(ok=True, username=user["username"], message="Signed in.")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """
    Predict churn probability for a single customer.

    Returns the probability, risk level, and top SHAP feature contributions.
    """
    _require_model()
    try:
        result = predictor.predict(customer.model_dump())
        return _build_response(result)
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict churn for a batch of up to 100 customers.
    """
    _require_model()
    if len(request.customers) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 customers per request.",
        )
    predictions = []
    for customer in request.customers:
        try:
            result = predictor.predict(customer.model_dump())
            predictions.append(_build_response(result))
        except Exception as exc:
            logger.error("Error on customer %s: %s", customer.customerID, exc)

    high_risk = sum(1 for p in predictions if p.risk_level == "High")

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=high_risk,
    )


@app.post("/predict-csv", response_model=CsvPredictionResponse, tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)) -> CsvPredictionResponse:
    """Score a customer CSV using the current model."""
    _require_model()
    raw = await _read_csv_upload(file)
    _validate_customer_columns(raw)
    if len(raw) > 500:
        raise HTTPException(status_code=400, detail="CSV scoring is limited to 500 rows per upload.")

    rows = []
    for index, record in raw.iterrows():
        customer = record.to_dict()
        customer.setdefault("customerID", f"ROW-{index + 1}")
        result = predictor.predict(customer)
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
    return CsvPredictionResponse(total=len(rows), high_risk_count=high_risk, rows=rows)


@app.post("/learning/upload", response_model=LearningUploadResponse, tags=["Learning"])
async def upload_learning_csv(file: UploadFile = File(...)) -> LearningUploadResponse:
    """
    Store labeled company CSV rows for future retraining.

    The file must include the normal customer columns plus a Churn column.
    """
    raw = await _read_csv_upload(file)
    _validate_customer_columns(raw)
    if "Churn" not in raw.columns:
        raise HTTPException(
            status_code=400,
            detail="Learning uploads need a Churn column with known Yes/No outcomes.",
        )
    accepted = raw[BATCH_REQUIRED_COLS + ["Churn"]].copy()
    accepted["source_file"] = file.filename
    accepted["uploaded_at"] = pd.Timestamp.utcnow().isoformat()

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FEEDBACK_PATH.exists():
        existing = pd.read_csv(FEEDBACK_PATH)
        stored = pd.concat([existing, accepted], ignore_index=True)
    else:
        stored = accepted
    stored.to_csv(FEEDBACK_PATH, index=False)

    return LearningUploadResponse(
        ok=True,
        accepted_rows=len(accepted),
        stored_rows=len(stored),
        message=(
            "Labeled rows saved for retraining. Run the retraining workflow after "
            "reviewing data quality and drift."
        ),
    )


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
