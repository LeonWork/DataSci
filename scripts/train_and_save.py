"""
scripts/train_and_save.py
-------------------------
Train the full churn prediction model and save artefacts to disk.

Run once after downloading the Telco CSV:
    python scripts/train_and_save.py

Outputs (in models/):
    churn_model.joblib    — fitted sklearn classifier
    churn_pipeline.joblib — fitted sklearn ColumnTransformer
    model_meta.json       — feature names + training metrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import joblib
import pandas as pd

from src.data.pipeline import (
    build_pipeline,
    prepare_dataframe,
)
from src.data.load_data import load_raw
from src.models.evaluate import compute_metrics
from src.models.train import train_logistic_regression
from src.api.storage import engine, learning_rows, company_schemas, get_company_schema, init_storage
from sqlalchemy import select
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


def load_training_data(company_id: str = "global") -> pd.DataFrame:
    datasets = []
    df_reference = None
    
    if company_id == "global":
        df_raw = load_raw()
        df_reference = df_raw
        datasets.append(("IBM Telco", df_raw))

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

        if FEEDBACK_PATH.exists():
            feedback = pd.read_csv(FEEDBACK_PATH)
            drop_cols = [c for c in ["source_file", "uploaded_at"] if c in feedback.columns]
            feedback = feedback.drop(columns=drop_cols)
            feedback = _align_training_columns(feedback, df_raw, source_prefix="FEEDBACK")
            logger.info("Including %d labeled company feedback rows", len(feedback))
            datasets.append(("company feedback", feedback))
        
    try:
        with engine().connect() as conn:
            query = select(learning_rows.c.row_json).where(learning_rows.c.status == "queued")
            if company_id != "global":
                query = query.where(learning_rows.c.company_id == company_id)
            rows = conn.execute(query).fetchall()
        if rows:
            db_records = [json.loads(r[0]) for r in rows]
            db_feedback = pd.DataFrame(db_records)
            if company_id == "global":
                db_feedback = _align_training_columns(db_feedback, df_reference, source_prefix="DB")
            logger.info("Including %d labeled rows from Postgres database queue", len(db_feedback))
            datasets.append(("postgres_queue", db_feedback))
    except Exception as e:
        logger.warning("Could not fetch database learning rows: %s", e)

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


def train_for_company(company_id: str) -> None:
    logger.info("Loading raw data for %s …", company_id)
    df_raw = load_training_data(company_id)

    if company_id == "global":
        logger.info("Preprocessing + feature engineering …")
        X, y = prepare_dataframe(df_raw)
    else:
        logger.info("Generic preprocessing for tenant %s …", company_id)
        # Drop metadata columns that aren't features
        drop_cols = [c for c in ["customerID", "source_file", "uploaded_at"] if c in df_raw.columns]
        df_clean = df_raw.drop(columns=drop_cols, errors="ignore")
        # Encode target
        if "Churn" in df_clean.columns:
            df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)
        # Fill numeric NaN with 0, categorical NaN with "Unknown"
        for col in df_clean.columns:
            if col == "Churn":
                continue
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna("Unknown")
        y = df_clean["Churn"]
        X = df_clean.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    schema = get_company_schema(company_id) if company_id != "global" else None
    if schema:
        num_cols = schema.get("numerical", [])
        cat_dict_or_list = schema.get("categorical", {})
        cat_cols = list(cat_dict_or_list.keys()) if isinstance(cat_dict_or_list, dict) else cat_dict_or_list
        # Only keep columns that actually exist in X
        num_cols = [c for c in num_cols if c in X.columns]
        cat_cols = [c for c in cat_cols if c in X.columns]
    else:
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]

    logger.info("Fitting sklearn pipeline …")
    pipe = build_pipeline(numerical_features=num_cols, categorical_features=cat_cols)
    X_train_t = pipe.fit_transform(X_train)
    X_test_t  = pipe.transform(X_test)

    logger.info("Training Logistic Regression model for production …")
    result = train_logistic_regression(
        X_train_t, y_train,
        X_test_t,  y_test,
        params={
            "max_iter": 2000,
            "C": 0.5,
            "class_weight": "balanced",
            "solver": "liblinear",
        },
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
    model_path    = MODELS_DIR / f"{company_id}_model.joblib"
    pipeline_path = MODELS_DIR / f"{company_id}_pipeline.joblib"
    meta_path     = MODELS_DIR / f"{company_id}_model_meta.json"

    joblib.dump(model, model_path)
    joblib.dump(pipe, pipeline_path)
    meta_path.write_text(json.dumps({
        "company_id":    company_id,
        "feature_names": feature_names,
        "metrics":       metrics,
        "num_features":  num_cols,
        "cat_features":  cat_cols,
        "model_family":  "LogisticRegression",
    }, indent=2))

    logger.info("Saved model      → %s", model_path)
    logger.info("Saved pipeline   → %s", pipeline_path)
    logger.info("Saved meta       → %s", meta_path)

    logger.info("\n=== Training Results ===")
    for k, v in metrics.items():
        logger.info("  %-12s %.4f", k, v)
    logger.info("\nRun the web app:  uvicorn src.api.main:app --reload --port 8000")


def main() -> None:
    init_storage()
    cid = sys.argv[1] if len(sys.argv) > 1 else "global"
    
    if cid == "all":
        companies = []
        try:
            with engine().connect() as conn:
                query = select(company_schemas.c.company_id)
                rows = conn.execute(query).fetchall()
                companies = [r[0] for r in rows]
        except Exception as e:
            logger.warning("Could not fetch company schemas: %s", e)
        
        logger.info("Found %d companies with custom schemas. Training all...", len(companies))
        for comp_id in companies:
            logger.info("--- Training for %s ---", comp_id)
            try:
                train_for_company(comp_id)
            except ValueError as e:
                if "No objects to concatenate" in str(e):
                    # No training data for this company — copy global model as fallback
                    import shutil
                    for suffix in ["_model.joblib", "_pipeline.joblib", "_model_meta.json"]:
                        src = MODELS_DIR / f"global{suffix}"
                        dst = MODELS_DIR / f"{comp_id}{suffix}"
                        if src.exists():
                            shutil.copy2(src, dst)
                    logger.info("No tenant data found — copied global model for %s", comp_id)
                else:
                    raise
    else:
        train_for_company(cid)


if __name__ == "__main__":
    main()
