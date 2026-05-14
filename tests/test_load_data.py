"""
tests/test_load_data.py
-----------------------
Unit tests for src/data/load_data.py

Run with:  pytest tests/ -v
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sample_df() -> pd.DataFrame:
    """Minimal fake Telco dataframe for testing."""
    return pd.DataFrame({
        "customerID":     ["0001-AAA", "0002-BBB", "0003-CCC"],
        "SeniorCitizen":  [0, 1, 0],
        "Partner":        ["Yes", "No", "Yes"],
        "Dependents":     ["No", "No", "Yes"],
        "tenure":         [1, 34, 2],
        "PhoneService":   ["No", "Yes", "Yes"],
        "Contract":       ["Month-to-month", "One year", "Month-to-month"],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges":   ["29.85", " ", "108.15"],   # note: whitespace string
        "Churn":          ["No", "No", "Yes"],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFixTypes:
    """Test the _fix_types helper applied inside load_raw."""

    def test_total_charges_coerced_to_float(self):
        from src.data.load_data import _fix_types
        df = _make_sample_df()
        result = _fix_types(df)
        assert result["TotalCharges"].dtype == float

    def test_whitespace_total_charges_becomes_nan(self):
        from src.data.load_data import _fix_types
        df = _make_sample_df()
        result = _fix_types(df)
        assert pd.isna(result.loc[1, "TotalCharges"])

    def test_senior_citizen_mapped_to_yes_no(self):
        from src.data.load_data import _fix_types
        df = _make_sample_df()
        result = _fix_types(df)
        assert set(result["SeniorCitizen"].unique()).issubset({"Yes", "No"})


class TestDataSummary:
    """Test data_summary utility."""

    def test_shape_returned(self):
        from src.data.load_data import data_summary, _fix_types
        df = _fix_types(_make_sample_df())
        summary = data_summary(df)
        assert summary["shape"] == (3, 10)

    def test_churn_rate_between_0_and_1(self):
        from src.data.load_data import data_summary, _fix_types
        df = _fix_types(_make_sample_df())
        summary = data_summary(df)
        assert 0.0 <= summary["churn_rate"] <= 1.0

    def test_missing_detected(self):
        from src.data.load_data import data_summary, _fix_types
        df = _fix_types(_make_sample_df())
        summary = data_summary(df)
        assert "TotalCharges" in summary["missing"]


class TestLoadRaw:
    """Test the main load_raw function with a mocked CSV."""

    def test_file_not_found_raises(self):
        from src.data.load_data import load_raw
        with pytest.raises(FileNotFoundError):
            load_raw(filepath="/nonexistent/path.csv")

    def test_loads_successfully_with_mock(self, tmp_path):
        import csv
        from src.data.load_data import load_raw

        # Write a minimal CSV to a tmp file
        csv_path = tmp_path / "mock_churn.csv"
        sample = _make_sample_df()
        sample.to_csv(csv_path, index=False)

        df = load_raw(filepath=csv_path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
