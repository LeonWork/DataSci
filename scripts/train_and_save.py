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


def main() -> None:
    logger.info("Loading raw data …")
    df_raw = load_raw()

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
    logger.info("\nRun the API:        uvicorn src.api.main:app --reload --port 8000")
    logger.info("Run the dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
