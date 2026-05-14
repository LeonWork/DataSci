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
import shap

from src.data.features import engineer_all_features
from src.data.preprocess import drop_ids, handle_missing
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH    = ROOT / "models" / "churn_model.joblib"
PIPELINE_PATH = ROOT / "models" / "churn_pipeline.joblib"
META_PATH     = ROOT / "models" / "model_meta.json"

MODEL_VERSION = "1.0.0"


class ChurnPredictor:
    """Wraps the sklearn pipeline + XGBoost model for inference."""

    def __init__(self) -> None:
        self.model = None
        self.pipeline = None
        self.feature_names: list[str] = []
        self.training_metrics: dict = {}
        self._loaded = False

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}.\n"
                "Run:  python scripts/train_and_save.py"
            )
        self.model    = joblib.load(MODEL_PATH)
        self.pipeline = joblib.load(PIPELINE_PATH)
        if META_PATH.exists():
            meta = json.loads(META_PATH.read_text())
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
        top_n: int = 5,
    ) -> dict:
        X = self._preprocess(raw)
        prob = float(self.model.predict_proba(X)[0, 1])
        pred = prob >= 0.5

        if prob < 0.30:
            risk = "Low"
        elif prob < 0.65:
            risk = "Medium"
        else:
            risk = "High"

        factors = self._shap_factors(X, top_n)

        return {
            "customer_id":       raw.get("customerID", "UNKNOWN"),
            "churn_probability": round(prob, 4),
            "churn_prediction":  pred,
            "risk_level":        risk,
            "top_factors":       factors,
            "model_version":     MODEL_VERSION,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }

    def _shap_factors(self, X: np.ndarray, top_n: int) -> list[dict]:
        try:
            explainer = shap.TreeExplainer(self.model)
            sv = explainer.shap_values(X)
            # Binary XGB: sv may be 2D (n_samples × n_features)
            if isinstance(sv, list):
                sv = sv[1]
            vals = sv[0]
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
            logger.warning("SHAP computation failed: %s", exc)
            return []


# Singleton — loaded once at startup
predictor = ChurnPredictor()
