"""
src/api/main.py
---------------
FastAPI application for the Customer Churn Prediction API.

Endpoints
---------
GET  /health          → liveness check
GET  /model-info      → model metadata and feature list
POST /predict         → single-customer churn prediction
POST /predict-batch   → batch predictions (up to 100 customers)

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.model_loader import predictor
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    FactorImpact,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
