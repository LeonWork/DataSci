"""
src/api/model_loader.py
-----------------------
Load the trained model and pipeline artefacts from disk.
Provides a predict() function for the FastAPI endpoints.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.features import engineer_all_features
from src.data.preprocess import drop_ids, handle_missing
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

MODEL_VERSION = "1.0.0"


class ChurnPredictor:
    """Wraps the sklearn preprocessing pipeline and trained classifier."""

    def __init__(self, company_id: str = "global") -> None:
        self.company_id = company_id
        self.model = None
        self.pipeline = None
        self.feature_names: list[str] = []
        self.training_metrics: dict = {}
        self._loaded = False
        
        self.model_path = MODELS_DIR / f"{company_id}_model.joblib"
        self.pipeline_path = MODELS_DIR / f"{company_id}_pipeline.joblib"
        self.meta_path = MODELS_DIR / f"{company_id}_model_meta.json"
        
        # Fallbacks for backwards compatibility
        if not self.model_path.exists() and company_id != "global":
            self.model_path = MODELS_DIR / "global_model.joblib"
            self.pipeline_path = MODELS_DIR / "global_pipeline.joblib"
            self.meta_path = MODELS_DIR / "global_model_meta.json"
            
        if not self.model_path.exists():
            self.model_path = MODELS_DIR / "churn_model.joblib"
            self.pipeline_path = MODELS_DIR / "churn_pipeline.joblib"
            self.meta_path = MODELS_DIR / "model_meta.json"

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found for {self.company_id} at {self.model_path}."
            )
        self.model    = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            self.feature_names    = meta.get("feature_names", [])
            self.training_metrics = meta.get("metrics", {})
        self._loaded = True
        logger.info("Model loaded ✓ — %d features", len(self.feature_names))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Preprocessing ──────────────────────────────────────────────────────

    def _preprocess(self, raw: dict) -> np.ndarray:
        df = pd.DataFrame([raw])
        df = handle_missing(df)
        df = engineer_all_features(df)
        df = drop_ids(df)
        # Remove target if accidentally included
        df = df.drop(columns=["Churn"], errors="ignore")
        return self.pipeline.transform(df)

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(
        self,
        raw: dict,
        top_n: int = 10,
    ) -> dict:
        X = self._preprocess(raw)
        prob = float(self.model.predict_proba(X)[0, 1])
        # Tuned threshold: Lowered from 0.5 to 0.35 to prioritize recall
        pred = prob >= 0.35

        if prob < 0.30:
            risk = "Low"
        elif prob < 0.65:
            risk = "Medium"
        else:
            risk = "High"

        factors = self._local_impact_factors(X, prob, top_n)

        return {
            "customer_id":       raw.get("customerID", "UNKNOWN"),
            "churn_probability": round(prob, 4),
            "churn_prediction":  pred,
            "risk_level":        risk,
            "top_factors":       factors,
            "model_version":     MODEL_VERSION,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }

    def _local_impact_factors(self, X: np.ndarray, base_prob: float, top_n: int) -> list[dict]:
        """
        Estimate local feature impact by zeroing one transformed feature at a time.

        This is model-agnostic and deploys without SHAP/XGBoost dependencies. A positive
        value means the current feature value is increasing churn probability relative
        to a neutralized version of that transformed feature.
        """
        try:
            X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X).copy()
            vals = []
            for i in range(X_dense.shape[1]):
                X_perturbed = X_dense.copy()
                X_perturbed[0, i] = 0
                perturbed_prob = float(self.model.predict_proba(X_perturbed)[0, 1])
                vals.append(base_prob - perturbed_prob)
            vals = np.asarray(vals)
            indices = np.argsort(np.abs(vals))[::-1][:top_n]
            factors = []
            for i in indices:
                name = self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
                v = float(vals[i])
                factors.append({
                    "feature":    name,
                    "shap_value": round(v, 4),
                    "direction":  "increases_risk" if v > 0 else "decreases_risk",
                })
            return factors
        except Exception as exc:
            logger.warning("Local impact computation failed: %s", exc)
            return []


class ModelRouter:
    def __init__(self) -> None:
        self.predictors: dict[str, ChurnPredictor] = {}

    def get_predictor(self, company_id: str) -> ChurnPredictor:
        if company_id not in self.predictors:
            predictor = ChurnPredictor(company_id)
            predictor.load()
            self.predictors[company_id] = predictor
        return self.predictors[company_id]

# Singleton router loaded once at startup
router = ModelRouter()
