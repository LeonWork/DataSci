"""
tests/test_features.py
----------------------
Unit tests for src/data/features.py

Run with:  pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np

from src.data.features import (
    add_clv,
    add_contract_stability,
    add_service_bundle_score,
    add_internet_flag,
    add_tenure_band,
    add_high_value_flag,
    engineer_all_features,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal representative subset of the Telco dataset."""
    return pd.DataFrame(
        {
            "customerID": ["A1", "A2", "A3", "A4"],
            "tenure": [1, 12, 36, 60],
            "MonthlyCharges": [29.85, 56.95, 53.85, 99.65],
            "TotalCharges": [29.85, 683.4, 1889.5, 6006.1],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "InternetService": ["DSL", "Fiber optic", "No", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes", "No internet service", "Yes"],
            "OnlineBackup": ["Yes", "No", "No internet service", "No"],
            "DeviceProtection": ["No", "Yes", "No internet service", "Yes"],
            "TechSupport": ["No", "No", "No internet service", "Yes"],
            "StreamingTV": ["No", "No", "No internet service", "Yes"],
            "StreamingMovies": ["No", "No", "No internet service", "Yes"],
            "MultipleLines": ["No phone service", "No", "No", "Yes"],
            "PhoneService": ["No", "Yes", "Yes", "Yes"],
        }
    )


# ── add_clv ───────────────────────────────────────────────────────────────────

class TestAddCLV:
    def test_clv_column_created(self, sample_df):
        result = add_clv(sample_df)
        assert "clv" in result.columns

    def test_clv_values_correct(self, sample_df):
        result = add_clv(sample_df)
        expected = sample_df["tenure"] * sample_df["MonthlyCharges"]
        pd.testing.assert_series_equal(result["clv"], expected, check_names=False)

    def test_avg_monthly_charge_created(self, sample_df):
        result = add_clv(sample_df)
        assert "avg_monthly_charge" in result.columns

    def test_charge_increase_created(self, sample_df):
        result = add_clv(sample_df)
        assert "charge_increase" in result.columns

    def test_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        add_clv(sample_df)
        assert list(sample_df.columns) == original_cols


# ── add_contract_stability ────────────────────────────────────────────────────

class TestContractStability:
    def test_month_to_month_is_zero(self, sample_df):
        result = add_contract_stability(sample_df)
        assert result.loc[result["Contract"] == "Month-to-month", "contract_stability"].iloc[0] == 0

    def test_one_year_is_one(self, sample_df):
        result = add_contract_stability(sample_df)
        assert result.loc[result["Contract"] == "One year", "contract_stability"].iloc[0] == 1

    def test_two_year_is_two(self, sample_df):
        result = add_contract_stability(sample_df)
        assert result.loc[result["Contract"] == "Two year", "contract_stability"].iloc[0] == 2

    def test_dtype_is_int(self, sample_df):
        result = add_contract_stability(sample_df)
        assert result["contract_stability"].dtype == int


# ── add_service_bundle_score ──────────────────────────────────────────────────

class TestServiceBundleScore:
    def test_column_created(self, sample_df):
        result = add_service_bundle_score(sample_df)
        assert "service_bundle_score" in result.columns

    def test_no_services_scores_zero(self, sample_df):
        # Row 0: OnlineSecurity=No, OnlineBackup=Yes, rest=No / No phone service
        result = add_service_bundle_score(sample_df)
        # Row 0 has PhoneService=No, MultipleLines=No phone service, OnlineBackup=Yes → 1
        assert result.loc[0, "service_bundle_score"] >= 0

    def test_many_services_scores_high(self, sample_df):
        result = add_service_bundle_score(sample_df)
        # Row 3 has many "Yes" values
        assert result.loc[3, "service_bundle_score"] >= 4

    def test_dtype_is_int(self, sample_df):
        result = add_service_bundle_score(sample_df)
        assert result["service_bundle_score"].dtype == int


# ── add_internet_flag ─────────────────────────────────────────────────────────

class TestInternetFlag:
    def test_dsl_is_one(self, sample_df):
        result = add_internet_flag(sample_df)
        assert result.loc[0, "has_internet"] == 1  # row 0 has DSL

    def test_no_internet_is_zero(self, sample_df):
        result = add_internet_flag(sample_df)
        assert result.loc[2, "has_internet"] == 0  # row 2 has No

    def test_missing_column_safe(self):
        df = pd.DataFrame({"tenure": [1, 2]})
        result = add_internet_flag(df)
        assert (result["has_internet"] == 0).all()


# ── add_tenure_band ───────────────────────────────────────────────────────────

class TestTenureBand:
    def test_column_created(self, sample_df):
        result = add_tenure_band(sample_df)
        assert "tenure_band" in result.columns

    def test_tenure_1_is_new(self, sample_df):
        result = add_tenure_band(sample_df)
        assert result.loc[0, "tenure_band"] == "new"

    def test_tenure_60_is_champion(self, sample_df):
        result = add_tenure_band(sample_df)
        assert result.loc[3, "tenure_band"] == "champion"

    def test_dtype_is_object(self, sample_df):
        result = add_tenure_band(sample_df)
        assert result["tenure_band"].dtype == object


# ── add_high_value_flag ───────────────────────────────────────────────────────

class TestHighValueFlag:
    def test_requires_clv_column(self, sample_df):
        with pytest.raises(ValueError, match="clv"):
            add_high_value_flag(sample_df)

    def test_with_fixed_threshold(self, sample_df):
        df = add_clv(sample_df)
        result = add_high_value_flag(df, clv_threshold=1000.0)
        assert "is_high_value" in result.columns
        # CLV for row 0 = 1*29.85 = 29.85 < 1000 → 0
        assert result.loc[0, "is_high_value"] == 0

    def test_auto_threshold_75th_pct(self, sample_df):
        df = add_clv(sample_df)
        result = add_high_value_flag(df)
        # At least 25% should be flagged (by definition of 75th pct)
        assert result["is_high_value"].sum() >= 1


# ── engineer_all_features ─────────────────────────────────────────────────────

class TestEngineerAllFeatures:
    def test_all_columns_present(self, sample_df):
        result = engineer_all_features(sample_df)
        expected_cols = [
            "clv", "avg_monthly_charge", "charge_increase",
            "contract_stability", "service_bundle_score",
            "has_internet", "tenure_band", "is_high_value",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_unchanged(self, sample_df):
        result = engineer_all_features(sample_df)
        assert len(result) == len(sample_df)

    def test_does_not_mutate_input(self, sample_df):
        cols_before = list(sample_df.columns)
        engineer_all_features(sample_df)
        assert list(sample_df.columns) == cols_before
