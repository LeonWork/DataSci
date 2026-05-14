"""
src/data/features.py
--------------------
Domain-driven feature engineering for the Telco Churn dataset.

New features created here:
    clv                  – Customer Lifetime Value (tenure × MonthlyCharges)
    avg_monthly_charge   – TotalCharges / max(tenure, 1)  (realised avg spend)
    charge_increase      – MonthlyCharges − avg_monthly_charge  (recent uplift)
    contract_stability   – Ordinal score: Month-to-month=0, One year=1, Two year=2
    service_bundle_score – Count of add-on services subscribed (0–8)
    has_internet         – 1 if InternetService != "No"
    tenure_band          – Bucketed tenure: new / mid / loyal / champion
    is_high_value        – 1 if clv > 75th-percentile of the training set
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Add-on services that count toward the bundle score
ADD_ON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "MultipleLines", "PhoneService",
]

CONTRACT_STABILITY_MAP = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}

TENURE_BANDS = [0, 12, 24, 48, np.inf]
TENURE_LABELS = ["new", "mid", "loyal", "champion"]


# ── Core feature functions ─────────────────────────────────────────────────────

def add_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customer Lifetime Value proxy: tenure (months) × MonthlyCharges.

    Also derives:
        avg_monthly_charge  = TotalCharges / max(tenure, 1)
        charge_increase     = MonthlyCharges − avg_monthly_charge
    """
    df = df.copy()
    df["clv"] = df["tenure"] * df["MonthlyCharges"]
    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_charge"]
    logger.debug("Added CLV features: clv, avg_monthly_charge, charge_increase")
    return df


def add_contract_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode Contract type as a stability score (0 / 1 / 2)."""
    df = df.copy()
    df["contract_stability"] = (
        df["Contract"].map(CONTRACT_STABILITY_MAP).fillna(0).astype(int)
    )
    logger.debug("Added contract_stability")
    return df


def add_service_bundle_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many add-on services the customer subscribes to.
    Columns with value "Yes" are counted; "No service" / "No" count as 0.
    """
    df = df.copy()
    present_cols = [c for c in ADD_ON_SERVICES if c in df.columns]
    df["service_bundle_score"] = (
        df[present_cols].eq("Yes").sum(axis=1).astype(int)
    )
    logger.debug(
        "Added service_bundle_score (cols: %s)", present_cols
    )
    return df


def add_internet_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: customer has any internet service (1) or not (0)."""
    df = df.copy()
    if "InternetService" in df.columns:
        df["has_internet"] = (df["InternetService"] != "No").astype(int)
    else:
        df["has_internet"] = 0
    logger.debug("Added has_internet flag")
    return df


def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket tenure into four lifecycle bands.

    new      : 0–12 months
    mid      : 13–24 months
    loyal    : 25–48 months
    champion : 49+ months
    """
    df = df.copy()
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=TENURE_BANDS,
        labels=TENURE_LABELS,
        right=True,
    ).astype(str)
    logger.debug("Added tenure_band")
    return df


def add_high_value_flag(
    df: pd.DataFrame,
    clv_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Binary flag for high-value customers (CLV above a threshold).

    Args:
        df:            DataFrame — must already contain the 'clv' column.
        clv_threshold: If None, use the 75th-percentile of the current batch.
                       Pass a fixed value (from training set) for test/prod.
    """
    df = df.copy()
    if "clv" not in df.columns:
        raise ValueError("Run add_clv() before add_high_value_flag().")
    threshold = clv_threshold if clv_threshold is not None else df["clv"].quantile(0.75)
    df["is_high_value"] = (df["clv"] >= threshold).astype(int)
    logger.debug("is_high_value threshold=%.2f", threshold)
    return df


# ── Convenience: run all feature steps ────────────────────────────────────────

def engineer_all_features(
    df: pd.DataFrame,
    clv_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.

    Args:
        df:            Raw or lightly preprocessed DataFrame.
        clv_threshold: Fixed CLV percentile for is_high_value (pass training value in prod).

    Returns:
        DataFrame with all new features appended.
    """
    df = add_clv(df)
    df = add_contract_stability(df)
    df = add_service_bundle_score(df)
    df = add_internet_flag(df)
    df = add_tenure_band(df)
    df = add_high_value_flag(df, clv_threshold=clv_threshold)
    logger.info(
        "Feature engineering complete — %d columns total", df.shape[1]
    )
    return df
