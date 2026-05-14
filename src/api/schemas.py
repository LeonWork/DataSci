"""
src/api/schemas.py
------------------
Pydantic request / response models for the Churn Prediction API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    customerID: str = Field(default="UNKNOWN")
    gender: Literal["Male", "Female"] = "Male"
    SeniorCitizen: Literal["Yes", "No"] = "No"
    Partner: Literal["Yes", "No"] = "No"
    Dependents: Literal["Yes", "No"] = "No"
    tenure: int = Field(default=1, ge=0, le=72)
    PhoneService: Literal["Yes", "No"] = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "DSL"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    MonthlyCharges: float = Field(default=29.85, ge=0, le=999)
    TotalCharges: float = Field(default=29.85, ge=0)


class FactorImpact(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: bool
    risk_level: Literal["Low", "Medium", "High"]
    top_factors: list[FactorImpact]
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    n_features: int
    feature_names: list[str]
    training_metrics: dict
    version: str
