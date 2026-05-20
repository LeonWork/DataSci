"""
scripts/generate_demo_data.py
-----------------------------
Generate a realistic synthetic Telco-schema dataset for demo purposes.
Produces ~7 000 rows that mirror the real IBM Telco Churn distribution.

Run:
    python scripts/generate_demo_data.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N   = 50000
OUT = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def _yn(p: float, n: int) -> np.ndarray:
    return RNG.choice(["Yes", "No"], size=n, p=[p, 1 - p])


def main() -> None:
    tenure = RNG.integers(0, 72, N).astype(float)

    # Contract — skewed toward month-to-month
    contract = RNG.choice(
        ["Month-to-month", "One year", "Two year"],
        size=N, p=[0.55, 0.21, 0.24],
    )

    internet = RNG.choice(
        ["DSL", "Fiber optic", "No"],
        size=N, p=[0.34, 0.44, 0.22],
    )

    monthly = np.where(
        internet == "No",
        RNG.uniform(20, 35, N),
        np.where(internet == "DSL", RNG.uniform(25, 65, N), RNG.uniform(55, 110, N)),
    ).round(2)

    # Simulate realistic TotalCharges (with 11 blanks for new customers)
    total = (monthly * tenure).round(2)
    blank_idx = RNG.choice(N, size=11, replace=False)
    total_str = total.astype(str)
    total_str[blank_idx] = " "

    # Internet-dependent add-ons
    def inet_col(p_yes: float) -> list[str]:
        vals = []
        for i in range(N):
            if internet[i] == "No":
                vals.append("No internet service")
            else:
                vals.append("Yes" if RNG.random() < p_yes else "No")
        return vals

    # Churn probability — depends on contract, tenure, internet
    churn_logit = (
        -1.5
        + (contract == "Month-to-month") * 2.2
        + (contract == "One year") * 0.5
        - tenure * 0.06
        + (internet == "Fiber optic") * 1.2
        + RNG.normal(0, 0.02, N)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = np.where(RNG.random(N) < churn_prob, "Yes", "No")

    phone = _yn(0.90, N)
    multi = []
    for i in range(N):
        if phone[i] == "No":
            multi.append("No phone service")
        else:
            multi.append("Yes" if RNG.random() < 0.42 else "No")

    df = pd.DataFrame({
        "customerID":      [f"{RNG.integers(1000,9999)}-{''.join(RNG.choice(list('ABCDEFGHIJ'), 5))}" for _ in range(N)],
        "gender":          RNG.choice(["Male", "Female"], N),
        "SeniorCitizen":   RNG.choice([0, 1], N, p=[0.84, 0.16]),
        "Partner":         _yn(0.48, N),
        "Dependents":      _yn(0.30, N),
        "tenure":          tenure.astype(int),
        "PhoneService":    phone,
        "MultipleLines":   multi,
        "InternetService": internet,
        "OnlineSecurity":  inet_col(0.29),
        "OnlineBackup":    inet_col(0.34),
        "DeviceProtection":inet_col(0.34),
        "TechSupport":     inet_col(0.29),
        "StreamingTV":     inet_col(0.38),
        "StreamingMovies": inet_col(0.39),
        "Contract":        contract,
        "PaperlessBilling":_yn(0.59, N),
        "PaymentMethod":   RNG.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            N, p=[0.34, 0.23, 0.22, 0.21],
        ),
        "MonthlyCharges":  monthly,
        "TotalCharges":    total_str,
        "Churn":           churn,
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"✓ Saved {N} rows → {OUT}")
    print(f"  Churn rate: {(churn == 'Yes').mean():.1%}")


if __name__ == "__main__":
    main()
