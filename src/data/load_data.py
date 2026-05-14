"""
src/data/load_data.py
---------------------
Utilities for loading the raw Telco Churn CSV and performing initial type casting.
"""

from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Default paths ────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_CSV = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw(filepath: Path | str = RAW_CSV) -> pd.DataFrame:
    """
    Load the raw Telco Churn CSV into a DataFrame.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Raw DataFrame with initial type fixes applied.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data not found at {filepath}.\n"
            "Run:  kaggle datasets download -d blastchar/telco-customer-churn "
            "-p data/raw --unzip"
        )

    logger.info("Loading raw data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    df = _fix_types(df)
    return df


def _fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply minimal type corrections to the raw DataFrame.

    - `TotalCharges` arrives as object; coerce to float (empty strings → NaN).
    - `SeniorCitizen` is 0/1 int; convert to Yes/No for consistency.
    """
    # TotalCharges: coerce whitespace-only strings to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.debug(
        "TotalCharges NaNs after coerce: %d", df["TotalCharges"].isna().sum()
    )

    # Normalise SeniorCitizen to string like other binary columns
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def data_summary(df: pd.DataFrame) -> dict:
    """
    Return a quick summary dict — useful in notebooks and tests.

    Returns:
        dict with keys: shape, dtypes, missing, churn_rate
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()

    churn_rate = (
        (df["Churn"] == "Yes").mean()
        if "Churn" in df.columns
        else None
    )

    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": missing,
        "churn_rate": round(churn_rate, 4) if churn_rate is not None else None,
    }
