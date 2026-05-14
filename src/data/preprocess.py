"""
src/data/preprocess.py
----------------------
Reusable preprocessing pipeline functions.

Week 2 will expand this into a full sklearn Pipeline.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns that will be dropped before modelling
COLS_TO_DROP = ["customerID"]

# Columns treated as binary categorical (Yes/No + unknown variants)
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "Churn", "SeniorCitizen",
]

# Multi-class categoricals
MULTI_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# Numerical columns (post type-fix)
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def drop_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID-like columns that carry no predictive signal."""
    cols = [c for c in COLS_TO_DROP if c in df.columns]
    logger.debug("Dropping columns: %s", cols)
    return df.drop(columns=cols)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values.

    Strategy:
        - TotalCharges NaN → 0 (new customers with 0 tenure)
        - No other columns have missings in the base dataset
    """
    n_before = df.isnull().sum().sum()
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    n_after = df.isnull().sum().sum()
    logger.info("Handled %d missing values → %d remaining", n_before, n_after)
    return df


def encode_target(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Encode the binary target column as 0 / 1."""
    df = df.copy()
    df[target_col] = (df[target_col] == "Yes").astype(int)
    logger.debug("Target encoded: %s", df[target_col].value_counts().to_dict())
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns.

    Binary (Yes/No) → 0/1
    Multi-class     → one-hot (drop_first=True to avoid multicollinearity)
    """
    df = df.copy()

    # Binary columns
    for col in BINARY_COLS:
        if col in df.columns and col != "Churn":
            df[col] = (df[col] == "Yes").astype(int)

    # Multi-class → one-hot
    multi_present = [c for c in MULTI_COLS if c in df.columns]
    if multi_present:
        df = pd.get_dummies(df, columns=multi_present, drop_first=True)
        logger.debug("One-hot encoded: %s", multi_present)

    return df


def get_X_y(df: pd.DataFrame, target_col: str = "Churn"):
    """
    Split DataFrame into feature matrix X and target vector y.

    Returns:
        X: pd.DataFrame
        y: pd.Series
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y
