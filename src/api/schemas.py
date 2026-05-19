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


class AdminSummaryResponse(BaseModel):
    storage_backend: str
    company_name: str
    total_predictions: int
    high_risk_predictions: int
    csv_upload_batches: int
    learning_rows_queued: int
    latest_upload_at: str | None = None
    model_type: str
    model_version: str
    model_auc: float | None = None
    retrain_recommended: bool


class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str
    invite_code: str | None = None


class AuthResponse(BaseModel):
    ok: bool
    username: str
    message: str
    access_token: str | None = None
    company_id: str | None = None
    role: str | None = None


class AuthConfigResponse(BaseModel):
    signup_enabled: bool
    signup_requires_invite: bool
    message: str


class CsvPredictionRow(BaseModel):
    customerID: str
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    top_driver: str
    tenure: int | None = None
    contract: str | None = None
    monthly_charges: float | None = None


class CsvPredictionResponse(BaseModel):
    total: int
    high_risk_count: int
    rows: list[CsvPredictionRow]


class LearningUploadResponse(BaseModel):
    ok: bool
    accepted_rows: int
    stored_rows: int
    message: str


class LearningStatusResponse(BaseModel):
    ok: bool
    storage_backend: str
    storage_path: str
    stored_rows: int
    churn_yes_count: int
    churn_no_count: int
    latest_uploaded_at: str | None = None
    retraining_command: str
    warning: str | None = None


class WorkspaceMemberResponse(BaseModel):
    username: str
    email: str = ""
    role: Literal["owner", "analyst", "viewer"]
    created_at: str
    last_seen_at: str | None = None


class WorkspaceResponse(BaseModel):
    company_id: str
    company_name: str
    plan: str
    status: str
    members: list[WorkspaceMemberResponse]


class CompanyOnboardRequest(BaseModel):
    company_id: str
    company_name: str
    owner_username: str
    owner_email: str
    owner_password: str


class CompanyOnboardResponse(BaseModel):
    ok: bool
    company_id: str
    message: str

