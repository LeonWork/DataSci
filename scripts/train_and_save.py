"""
scripts/train_and_save.py
-------------------------
Train the full churn prediction model and save artefacts to disk.

Run once after downloading the Telco CSV:
    python scripts/train_and_save.py

Outputs (in models/):
    churn_model.joblib    — fitted XGBClassifier
    churn_pipeline.joblib — fitted sklearn ColumnTransformer
    model_meta.json       — feature names + training metrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.data.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_pipeline,
    prepare_dataframe,
)
from src.data.load_data import load_raw
from src.models.evaluate import compute_metrics
from src.models.train import train_xgboost
from src.utils.logger import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger("train_and_save")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
FEEDBACK_PATH = ROOT / "data" / "company_feedback.csv"
EXTERNAL_LABELED_DIR = ROOT / "data" / "external" / "labeled_churn"
EXTERNAL_COLUMN_MAP = {
    "Customer ID": "customerID",
    "CustomerID": "customerID",
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen",
    "Tenure": "tenure",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Label": "Churn",
    "Churn Value": "Churn",
}


def load_training_data() -> pd.DataFrame:
    df_raw = load_raw()
    datasets = [("IBM Telco", df_raw)]

    if EXTERNAL_LABELED_DIR.exists():
        for csv_path in sorted(EXTERNAL_LABELED_DIR.glob("*.csv")):
            external = pd.read_csv(csv_path)
            try:
                external = _align_training_columns(external, df_raw, source_prefix=csv_path.stem)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", csv_path, exc)
                continue
            logger.info("Including %d rows from %s", len(external), csv_path)
            datasets.append((csv_path.name, external))

    if not FEEDBACK_PATH.exists():
        logger.info("Training sources: %s", ", ".join(name for name, _ in datasets))
        return pd.concat([df for _, df in datasets], ignore_index=True)

    feedback = pd.read_csv(FEEDBACK_PATH)
    drop_cols = [c for c in ["source_file", "uploaded_at"] if c in feedback.columns]
    feedback = feedback.drop(columns=drop_cols)
    feedback = _align_training_columns(feedback, df_raw, source_prefix="FEEDBACK")
    logger.info("Including %d labeled company feedback rows", len(feedback))
    datasets.append(("company feedback", feedback))
    logger.info("Training sources: %s", ", ".join(name for name, _ in datasets))
    return pd.concat([df for _, df in datasets], ignore_index=True)


def _align_training_columns(df: pd.DataFrame, reference: pd.DataFrame, source_prefix: str) -> pd.DataFrame:
    df = _normalize_external_columns(df)
    if "customerID" not in df.columns and "customerID" in reference.columns:
        df = df.copy()
        df.insert(0, "customerID", [f"{source_prefix}-{i + 1}" for i in range(len(df))])
    if set(reference.columns).issubset(df.columns):
        return df[reference.columns]
    missing = [c for c in reference.columns if c not in df.columns]
    raise ValueError(f"Additional training data is missing columns: {missing}")


def _normalize_external_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in EXTERNAL_COLUMN_MAP.items() if k in df.columns}).copy()
    df = df.loc[:, ~df.columns.duplicated()]
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({
            1: "Yes",
            0: "No",
            "1": "Yes",
            "0": "No",
            "Yes": "Yes",
            "No": "No",
            "True": "Yes",
            "False": "No",
            True: "Yes",
            False: "No",
        }).fillna(df["Churn"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({
            1: "Yes",
            0: "No",
            "1": "Yes",
            "0": "No",
            "Yes": "Yes",
            "No": "No",
            True: "Yes",
            False: "No",
        }).fillna(df["SeniorCitizen"])
    return df


def main() -> None:
    logger.info("Loading raw data …")
    df_raw = load_training_data()

    logger.info("Preprocessing + feature engineering …")
    X, y = prepare_dataframe(df_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    logger.info("Fitting sklearn pipeline …")
    pipe = build_pipeline(numerical_features=num_cols, categorical_features=cat_cols)
    X_train_t = pipe.fit_transform(X_train)
    X_test_t  = pipe.transform(X_test)

    logger.info("Training XGBoost model …")
    result = train_xgboost(
        X_train_t, y_train,
        X_test_t,  y_test,
        log_to_mlflow=False,
    )
    model   = result["model"]
    metrics = result["metrics"]

    # Feature names from pipeline
    try:
        feature_names = pipe.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

    # Save artefacts
    model_path    = MODELS_DIR / "churn_model.joblib"
    pipeline_path = MODELS_DIR / "churn_pipeline.joblib"
    meta_path     = MODELS_DIR / "model_meta.json"

    joblib.dump(model, model_path)
    joblib.dump(pipe, pipeline_path)
    meta_path.write_text(json.dumps({
        "feature_names": feature_names,
        "metrics":       metrics,
        "num_features":  num_cols,
        "cat_features":  cat_cols,
    }, indent=2))

    logger.info("Saved model      → %s", model_path)
    logger.info("Saved pipeline   → %s", pipeline_path)
    logger.info("Saved meta       → %s", meta_path)

    logger.info("\n=== Training Results ===")
    for k, v in metrics.items():
        logger.info("  %-12s %.4f", k, v)
    logger.info("\nRun the web app:  uvicorn src.api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
