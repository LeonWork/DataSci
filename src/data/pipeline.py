"""
src/data/pipeline.py
--------------------
Full sklearn-compatible preprocessing + feature engineering Pipeline.

Usage
-----
    from src.data.pipeline import build_pipeline, run_full_pipeline

    # Training
    pipe = build_pipeline()
    X_train_t = pipe.fit_transform(X_train)

    # Inference
    X_test_t = pipe.transform(X_test)

    # Or use the convenience wrapper that loads raw data → ready arrays:
    X_train, X_test, y_train, y_test = run_full_pipeline()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.data.load_data import load_raw
from src.data.features import engineer_all_features
from src.data.preprocess import (
    drop_ids,
    handle_missing,
    encode_target,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Column groups ─────────────────────────────────────────────────────────────
#   Derived after feature engineering; updated if schema changes.

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "clv",
    "avg_monthly_charge",
    "charge_increase",
    "contract_stability",
    "service_bundle_score",
    "has_internet",
    "is_high_value",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_band",
    "SeniorCitizen",
]

TARGET_COL = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ── Transformer blocks ────────────────────────────────────────────────────────

def _numerical_transformer() -> Pipeline:
    """Impute → scale numerical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _categorical_transformer() -> Pipeline:
    """Impute → one-hot encode categorical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                ),
            ),
        ]
    )


# ── Public API ────────────────────────────────────────────────────────────────

def build_pipeline(
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that scales numerics and
    one-hot encodes categoricals.

    Args:
        numerical_features:  Override default numerical column list.
        categorical_features: Override default categorical column list.

    Returns:
        Unfitted sklearn ColumnTransformer.
    """
    num_cols = numerical_features or NUMERICAL_FEATURES
    cat_cols = categorical_features or CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", _numerical_transformer(), num_cols),
            ("cat", _categorical_transformer(), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    logger.debug(
        "Pipeline built — %d numerical + %d categorical features",
        len(num_cols),
        len(cat_cols),
    )
    return preprocessor


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run all pre-sklearn steps: clean → feature engineer → split X/y.

    Args:
        df: Raw DataFrame from load_raw().

    Returns:
        (X, y) where X is the feature DataFrame and y is the binary target.
    """
    df = handle_missing(df)
    df = engineer_all_features(df)
    df = encode_target(df, target_col=TARGET_COL)
    df = drop_ids(df)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def run_full_pipeline(
    filepath: Path | str | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    End-to-end convenience function:
        raw CSV → cleaned → feature-engineered → split → transformed.

    Args:
        filepath:     Path to the raw CSV (defaults to data/raw/).
        test_size:    Fraction of data for the test set.
        random_state: Reproducibility seed.

    Returns:
        (X_train_t, X_test_t, y_train, y_test, fitted_pipeline)

        X_train_t / X_test_t are dense NumPy arrays ready for modelling.
    """
    kwargs = {"filepath": filepath} if filepath else {}
    df = load_raw(**kwargs)
    X, y = prepare_dataframe(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Only keep columns that actually exist in X
    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pipe = build_pipeline(
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    X_train_t = pipe.fit_transform(X_train)
    X_test_t = pipe.transform(X_test)

    logger.info(
        "Pipeline complete — train: %s | test: %s | features: %d",
        X_train_t.shape,
        X_test_t.shape,
        X_train_t.shape[1],
    )
    return X_train_t, X_test_t, y_train, y_test, pipe
