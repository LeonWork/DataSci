"""
tests/test_preprocess.py
------------------------
Unit tests for src/data/preprocess.py
"""

import pytest
import pandas as pd
from src.data.preprocess import (
    drop_ids,
    handle_missing,
    encode_target,
    encode_categoricals,
    get_X_y,
)


def _base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID":     ["A", "B"],
        "SeniorCitizen":  ["No", "Yes"],
        "Partner":        ["Yes", "No"],
        "Dependents":     ["No", "No"],
        "tenure":         [1, 24],
        "PhoneService":   ["Yes", "Yes"],
        "PaperlessBilling": ["Yes", "No"],
        "MultipleLines":  ["No phone service", "No"],
        "InternetService":["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup":   ["Yes", "No"],
        "DeviceProtection":["No", "Yes"],
        "TechSupport":    ["No", "No"],
        "StreamingTV":    ["No", "Yes"],
        "StreamingMovies":["No", "No"],
        "Contract":       ["Month-to-month", "Two year"],
        "PaymentMethod":  ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 79.65],
        "TotalCharges":   [29.85, None],
        "Churn":          ["No", "Yes"],
    })


class TestDropIds:
    def test_customer_id_removed(self):
        df = _base_df()
        result = drop_ids(df)
        assert "customerID" not in result.columns

    def test_other_cols_intact(self):
        df = _base_df()
        result = drop_ids(df)
        assert "tenure" in result.columns


class TestHandleMissing:
    def test_total_charges_nan_filled(self):
        df = _base_df()
        result = handle_missing(df)
        assert result["TotalCharges"].isna().sum() == 0

    def test_no_remaining_nulls(self):
        df = _base_df()
        result = handle_missing(df)
        assert result.isnull().sum().sum() == 0


class TestEncodeTarget:
    def test_yes_becomes_1(self):
        df = _base_df()
        result = encode_target(df)
        assert result.loc[result.index[1], "Churn"] == 1

    def test_no_becomes_0(self):
        df = _base_df()
        result = encode_target(df)
        assert result.loc[result.index[0], "Churn"] == 0


class TestGetXy:
    def test_y_is_series(self):
        df = encode_target(_base_df())
        X, y = get_X_y(df)
        assert isinstance(y, pd.Series)

    def test_churn_not_in_X(self):
        df = encode_target(_base_df())
        X, y = get_X_y(df)
        assert "Churn" not in X.columns
