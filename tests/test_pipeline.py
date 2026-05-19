"""
tests/test_pipeline.py
----------------------
Unit tests for src/data/pipeline.py

Tests focus on the pipeline building and prepare_dataframe functions,
using synthetic data so no CSV is required.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.data.pipeline import (
    build_pipeline,
    prepare_dataframe,
)

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "clv", "contract_stability", "service_bundle_score", "has_internet"]
CATEGORICAL_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def full_sample_df() -> pd.DataFrame:
    """
    A 20-row synthetic DataFrame matching the Telco schema.
    Includes all columns needed by prepare_dataframe().
    """
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice(["No", "Yes"], n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(1, 72, n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
            ),
            "MonthlyCharges": rng.uniform(20, 110, n).round(2),
            "TotalCharges": rng.uniform(20, 8000, n).round(2),
            "Churn": rng.choice(["Yes", "No"], n),
        }
    )


# ── build_pipeline ────────────────────────────────────────────────────────────

class TestBuildPipeline:
    def test_returns_column_transformer(self, full_sample_df):
        from sklearn.compose import ColumnTransformer
        pipe = build_pipeline(numerical_features=[], categorical_features=[])
        assert isinstance(pipe, ColumnTransformer)

    def test_has_two_transformers(self, full_sample_df):
        pipe = build_pipeline(numerical_features=[], categorical_features=[])
        names = [t[0] for t in pipe.transformers]
        assert "num" in names
        assert "cat" in names

    def test_custom_feature_lists(self):
        pipe = build_pipeline(
            numerical_features=["tenure"],
            categorical_features=["Contract"],
        )
        num_cols = pipe.transformers[0][2]
        cat_cols = pipe.transformers[1][2]
        assert num_cols == ["tenure"]
        assert cat_cols == ["Contract"]

    def test_fit_transform_runs(self, full_sample_df):
        from src.data.features import engineer_all_features
        from src.data.preprocess import drop_ids, handle_missing, encode_target
        df = handle_missing(full_sample_df)
        df = engineer_all_features(df)
        df = encode_target(df)
        df = drop_ids(df)

        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        pipe = build_pipeline(numerical_features=num_cols, categorical_features=cat_cols)

        X = df.drop(columns=["Churn"])
        result = pipe.fit_transform(X)
        assert result.shape[0] == len(df)
        assert result.shape[1] > 0


# ── prepare_dataframe ─────────────────────────────────────────────────────────

class TestPrepareDataframe:
    def test_returns_x_and_y(self, full_sample_df):
        X, y = prepare_dataframe(full_sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_churn_not_in_X(self, full_sample_df):
        X, y = prepare_dataframe(full_sample_df)
        assert "Churn" not in X.columns

    def test_customerid_not_in_X(self, full_sample_df):
        X, y = prepare_dataframe(full_sample_df)
        assert "customerID" not in X.columns

    def test_y_is_binary(self, full_sample_df):
        _, y = prepare_dataframe(full_sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_engineered_features_present(self, full_sample_df):
        X, _ = prepare_dataframe(full_sample_df)
        for col in ["clv", "contract_stability", "service_bundle_score", "has_internet"]:
            assert col in X.columns, f"Missing feature: {col}"

    def test_row_count_preserved(self, full_sample_df):
        X, y = prepare_dataframe(full_sample_df)
        assert len(X) == len(full_sample_df)
        assert len(y) == len(full_sample_df)

    def test_no_missing_in_y(self, full_sample_df):
        _, y = prepare_dataframe(full_sample_df)
        assert y.isna().sum() == 0
