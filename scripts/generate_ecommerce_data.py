import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

RNG = np.random.default_rng(123)
N   = 5000
OUT = ROOT / "data" / "external" / "ecommerce_churn.csv"

def main() -> None:
    # Schema completely different from Telco
    
    # Numerical
    days_since_last_purchase = RNG.integers(1, 365, N).astype(float)
    total_spent = RNG.uniform(10, 5000, N).round(2)
    support_tickets = RNG.integers(0, 10, N).astype(int)
    website_visits_last_month = RNG.integers(0, 50, N).astype(int)
    
    # Categorical
    subscription_tier = RNG.choice(["Free", "Basic", "Premium"], size=N, p=[0.5, 0.3, 0.2])
    payment_method = RNG.choice(["Credit Card", "PayPal", "Apple Pay", "Crypto"], size=N)
    country = RNG.choice(["USA", "Canada", "UK", "Australia", "Germany"], size=N)
    device_type = RNG.choice(["Mobile", "Desktop", "Tablet"], size=N)
    
    # Churn probability logic
    churn_logit = (
        -2.0
        + (days_since_last_purchase > 180) * 1.5
        + (support_tickets > 3) * 1.2
        - (subscription_tier == "Premium") * 1.0
        - (website_visits_last_month > 20) * 0.8
        + RNG.normal(0, 0.5, N)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = np.where(RNG.random(N) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID":      [f"ECOMM-{RNG.integers(10000,99999)}" for _ in range(N)],
        "days_since_last_purchase": days_since_last_purchase,
        "total_spent": total_spent,
        "support_tickets": support_tickets,
        "website_visits_last_month": website_visits_last_month,
        "subscription_tier": subscription_tier,
        "payment_method": payment_method,
        "country": country,
        "device_type": device_type,
        "Churn": churn,
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"✓ Saved {N} rows → {OUT}")
    print(f"  Churn rate: {(churn == 'Yes').mean():.1%}")

if __name__ == "__main__":
    main()
