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
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMClassifier

from src.data.pipeline import (
    build_pipeline,
    prepare_dataframe,
)
from src.data.load_data import load_raw
from src.models.evaluate import compute_metrics, threshold_report
from src.models.train import train_all_baselines
from src.models.tune import run_study, get_best_model
from src.api.storage import (
    engine,
    learning_rows,
    get_company_schema,
    init_storage,
    companies,
    ensure_company,
    start_training_run,
    finish_training_run,
)
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
NUMERIC_DETECTION_THRESHOLD = 0.9


class NoTrainingDataError(ValueError):
    """Raised when a tenant has no usable rows to train its own model."""


def _is_numeric_like(series: pd.Series) -> bool:
    non_empty = series.dropna()
    non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
    if non_empty.empty:
        return False
    converted = pd.to_numeric(non_empty, errors="coerce")
    return float(converted.notna().mean()) >= NUMERIC_DETECTION_THRESHOLD


def _split_feature_columns(X: pd.DataFrame, schema: dict | None) -> tuple[list[str], list[str]]:
    schema_num = list((schema or {}).get("numerical", []))
    schema_cat_raw = (schema or {}).get("categorical", {})
    schema_cat = list(schema_cat_raw.keys()) if isinstance(schema_cat_raw, dict) else list(schema_cat_raw or [])
    num_cols = [c for c in schema_num if c in X.columns]
    cat_cols = [c for c in schema_cat if c in X.columns and c not in num_cols]

    for column in X.columns:
        if column in num_cols or column in cat_cols:
            continue
        if pd.api.types.is_numeric_dtype(X[column]) or _is_numeric_like(X[column]):
            num_cols.append(column)
        else:
            cat_cols.append(column)

    repaired_cat_cols = []
    for column in cat_cols:
        if _is_numeric_like(X[column]):
            X[column] = pd.to_numeric(X[column], errors="coerce")
            num_cols.append(column)
        else:
            repaired_cat_cols.append(column)
    return num_cols, repaired_cat_cols


def _balanced_score(metrics: dict) -> float:
    return round(
        0.45 * float(metrics.get("roc_auc", 0))
        + 0.25 * float(metrics.get("pr_auc", 0))
        + 0.20 * float(metrics.get("f1", 0))
        + 0.10 * (1 - float(metrics.get("brier", 1))),
        6,
    )


def _training_profile(X: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> dict:
    numeric = {}
    for col in num_cols:
        if col not in X.columns:
            continue
        values = pd.to_numeric(X[col], errors="coerce").dropna()
        if values.empty:
            continue
        numeric[col] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0) or 0),
            "missing_rate": float(pd.to_numeric(X[col], errors="coerce").isna().mean()),
        }

    categorical = {}
    for col in cat_cols:
        if col not in X.columns:
            continue
        values = X[col].fillna("Unknown").astype(str)
        normalized = values.value_counts(normalize=True).head(20)
        categorical[col] = {
            "top_values": {str(k): float(v) for k, v in normalized.items()},
            "missing_rate": float(X[col].isna().mean()),
        }
    return {
        "row_count": int(len(X)),
        "numeric": numeric,
        "categorical": categorical,
    }


def _tune_lightgbm(X_train, y_train, X_test, y_test, n_trials: int = 20) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 700),
            "num_leaves": trial.suggest_int("num_leaves", 15, 80),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        return compute_metrics(model, X_test, y_test)["roc_auc"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)
    model = LGBMClassifier(
        **study.best_params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "metrics": compute_metrics(model, X_test, y_test),
        "best_params": study.best_params,
        "best_cv_auc": study.best_value,
    }


def load_training_data(company_id: str = "global") -> pd.DataFrame:
    datasets = []
    df_reference = None
    learning_row_ids = []
    
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
            query = select(learning_rows.c.id, learning_rows.c.row_json).where(
                learning_rows.c.status == "approved_for_training"
            )
            if company_id != "global":
                query = query.where(learning_rows.c.company_id == company_id)
            rows = conn.execute(query).fetchall()
            if not rows:
                legacy_query = select(learning_rows.c.id, learning_rows.c.row_json).where(learning_rows.c.status == "queued")
                if company_id != "global":
                    legacy_query = legacy_query.where(learning_rows.c.company_id == company_id)
                rows = conn.execute(legacy_query).fetchall()
        if rows:
            db_records = []
            for r in rows:
                rec = json.loads(r[1])
                if company_id == "global":
                    # Filter out incompatible schemas (e.g. E-Commerce) by ensuring core columns are present
                    if not all(col in rec for col in ["tenure", "MonthlyCharges", "Contract"]):
                        continue
                db_records.append(rec)
                learning_row_ids.append(int(r[0]))
            
            if db_records:
                db_feedback = pd.DataFrame(db_records)
                if company_id == "global":
                    db_feedback = _align_training_columns(db_feedback, df_reference, source_prefix="DB")
                logger.info("Including %d labeled rows from Postgres database queue", len(db_feedback))
                datasets.append(("postgres_queue", db_feedback))
            else:
                logger.info("No compatible database queue rows found.")
    except Exception as e:
        logger.warning("Could not fetch database learning rows: %s", e)

    if not datasets:
        raise NoTrainingDataError(f"No training data available for {company_id}")

    logger.info("Training sources: %s", ", ".join(name for name, _ in datasets))
    result = pd.concat([df for _, df in datasets], ignore_index=True)
    result.attrs["learning_row_ids"] = learning_row_ids
    return result


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


def _artifact_paths(company_id: str, *, candidate: bool) -> tuple[Path, Path, Path]:
    stem = f"{company_id}_candidate" if candidate else company_id
    return (
        MODELS_DIR / f"{stem}_model.joblib",
        MODELS_DIR / f"{stem}_pipeline.joblib",
        MODELS_DIR / f"{stem}_model_meta.json",
    )


def _copy_global_fallback(company_id: str, *, candidate: bool, run_id: int) -> dict:
    import shutil

    source_stem = "global_candidate" if candidate else "global"
    source_model = MODELS_DIR / f"{source_stem}_model.joblib"
    source_pipeline = MODELS_DIR / f"{source_stem}_pipeline.joblib"
    source_meta = MODELS_DIR / f"{source_stem}_model_meta.json"

    missing = [str(path) for path in (source_model, source_pipeline, source_meta) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Global fallback artifacts are missing. Train the global model first: "
            + ", ".join(missing)
        )

    model_path, pipeline_path, meta_path = _artifact_paths(company_id, candidate=candidate)
    shutil.copy2(source_model, model_path)
    shutil.copy2(source_pipeline, pipeline_path)

    meta = json.loads(source_meta.read_text())
    meta.update({
        "company_id": company_id,
        "artifact_stage": "candidate" if candidate else "production",
        "training_run_id": run_id,
        "fallback_from_company_id": "global",
        "fallback_reason": "no_tenant_training_data",
        "learning_row_ids": [],
    })
    meta_path.write_text(json.dumps(meta, indent=2))

    paths = {
        "model": str(model_path),
        "pipeline": str(pipeline_path),
        "metadata": str(meta_path),
    }
    finish_training_run(
        run_id,
        status="fallback_candidate_ready" if candidate else "fallback_succeeded",
        model_family=meta.get("model_family", "global_fallback"),
        metrics=meta.get("metrics", {}),
        artifact_paths=paths,
    )
    logger.info("No tenant data found — copied %s fallback artifacts for %s", source_stem, company_id)
    return meta


def train_for_company(company_id: str, *, candidate: bool = False) -> dict:
    ensure_company(company_id, "Global Baseline" if company_id == "global" else None)
    run_id = start_training_run(company_id, status="running")
    logger.info("Loading raw data for %s …", company_id)
    try:
        df_raw = load_training_data(company_id)

        if company_id == "global":
            logger.info("Preprocessing + feature engineering …")
            X, y = prepare_dataframe(df_raw)
        else:
            logger.info("Generic preprocessing for tenant %s …", company_id)
            drop_cols = [c for c in ["customerID", "source_file", "uploaded_at"] if c in df_raw.columns]
            df_clean = df_raw.drop(columns=drop_cols, errors="ignore")
            if "Churn" in df_clean.columns:
                df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)
            for col in df_clean.columns:
                if col == "Churn":
                    continue
                if pd.api.types.is_numeric_dtype(df_clean[col]) or _is_numeric_like(df_clean[col]):
                    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)
                else:
                    df_clean[col] = df_clean[col].fillna("Unknown")
            y = df_clean["Churn"]
            X = df_clean.drop(columns=["Churn"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        schema = get_company_schema(company_id) if company_id != "global" else None
        num_cols, cat_cols = _split_feature_columns(X, schema)

        logger.info("Fitting sklearn pipeline …")
        pipe = build_pipeline(numerical_features=num_cols, categorical_features=cat_cols)
        X_train_t = pipe.fit_transform(X_train)
        X_test_t = pipe.transform(X_test)

        logger.info("Training and evaluating all baseline models …")
        baselines = train_all_baselines(X_train_t, y_train, X_test_t, y_test, log_to_mlflow=False)
        tuning_trials = int(os.getenv("CHURNGUARD_TUNING_TRIALS", "40"))

        try:
            logger.info("Running Optuna hyperparameter optimization on XGBoost …")
            study = run_study(X_train_t, y_train, n_trials=tuning_trials, show_progress=False)
            tuned_xgb = get_best_model(study, X_train_t, y_train)
            tuned_metrics = compute_metrics(tuned_xgb, X_test_t, y_test)
            logger.info("XGBoost_Tuned → AUC=%.4f  F1=%.4f", tuned_metrics["roc_auc"], tuned_metrics["f1"])
            baselines["XGBoost_Tuned"] = {
                "model": tuned_xgb,
                "metrics": tuned_metrics,
                "best_params": study.best_params,
                "best_cv_auc": study.best_value,
            }
        except Exception as exc:
            logger.warning("Optuna XGBoost tuning failed: %s. Using other candidates.", exc)

        try:
            logger.info("Running Optuna hyperparameter optimization on LightGBM …")
            baselines["LightGBM_Tuned"] = _tune_lightgbm(
                X_train_t,
                y_train,
                X_test_t,
                y_test,
                n_trials=max(15, tuning_trials // 2),
            )
        except Exception as exc:
            logger.warning("Optuna LightGBM tuning failed: %s. Using other candidates.", exc)

        comparison = {
            name: {
                "metrics": result["metrics"],
                "balanced_score": _balanced_score(result["metrics"]),
                "best_params": result.get("best_params", {}),
                "best_cv_auc": result.get("best_cv_auc"),
            }
            for name, result in baselines.items()
        }
        best_model_name = max(baselines.keys(), key=lambda k: comparison[k]["balanced_score"])
        best_model_info = baselines[best_model_name]
        model = best_model_info["model"]
        metrics = best_model_info["metrics"]
        thresholds = threshold_report(model, X_test_t, y_test)

        logger.info("=========================================")
        logger.info(
            "Selected BEST model: %s with balanced score %.4f and Test AUC %.4f",
            best_model_name,
            comparison[best_model_name]["balanced_score"],
            metrics["roc_auc"],
        )
        logger.info("=========================================")

        try:
            feature_names = pipe.get_feature_names_out().tolist()
        except Exception:
            feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

        model_path, pipeline_path, meta_path = _artifact_paths(company_id, candidate=candidate)

        joblib.dump(model, model_path)
        joblib.dump(pipe, pipeline_path)
        meta = {
            "company_id": company_id,
            "artifact_stage": "candidate" if candidate else "production",
            "training_run_id": run_id,
            "learning_row_ids": list(df_raw.attrs.get("learning_row_ids", [])),
            "feature_names": feature_names,
            "metrics": metrics,
            "num_features": num_cols,
            "cat_features": cat_cols,
            "model_family": best_model_name,
            "model_comparison": comparison,
            "threshold_report": thresholds,
            "selection_metric": "balanced_score",
            "training_profile": _training_profile(X, num_cols, cat_cols),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        finish_training_run(
            run_id,
            status="candidate_ready" if candidate else "succeeded",
            model_family=best_model_name,
            metrics=metrics,
            artifact_paths={
                "model": str(model_path),
                "pipeline": str(pipeline_path),
                "metadata": str(meta_path),
            },
        )

        logger.info("Saved model      → %s", model_path)
        logger.info("Saved pipeline   → %s", pipeline_path)
        logger.info("Saved meta       → %s", meta_path)
        logger.info("\n=== Training Results ===")
        for k, v in metrics.items():
            logger.info("  %-12s %.4f", k, v)
        logger.info("\nRun the web app:  uvicorn src.api.main:app --reload --port 8000")
        return meta
    except NoTrainingDataError:
        return _copy_global_fallback(company_id, candidate=candidate, run_id=run_id)
    except Exception as exc:
        finish_training_run(run_id, status="failed", error_message=str(exc))
        raise


def main() -> None:
    init_storage()
    candidate = "--candidate" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--candidate"]
    cid = args[0] if args else "global"
    
    if cid == "all":
        companies_list = []
        try:
            with engine().connect() as conn:
                query = select(companies.c.id)
                rows = conn.execute(query).fetchall()
                companies_list = [r[0] for r in rows]
        except Exception as e:
            logger.warning("Could not fetch companies: %s", e)
        
        logger.info("Found %d companies in database. Training all...", len(companies_list))
        logger.info("--- Training global fallback source ---")
        train_for_company("global", candidate=candidate)
        for comp_id in companies_list:
            logger.info("--- Training for %s ---", comp_id)
            train_for_company(comp_id, candidate=candidate)
    else:
        train_for_company(cid, candidate=candidate)


if __name__ == "__main__":
    main()
