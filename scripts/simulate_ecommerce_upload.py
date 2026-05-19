"""
Simulate uploading ecommerce churn data for the acme_ecommerce tenant.
Inserts rows directly into the learning_rows table.
"""
import sys, json, os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env so DATABASE_URL points to Postgres
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd
from src.api.storage import engine, ensure_company
from sqlalchemy import text

def main() -> None:
    company_id = "acme_ecommerce"
    ensure_company(company_id, "Acme E-Commerce")

    df = pd.read_csv(ROOT / "data" / "external" / "ecommerce_churn.csv")
    print(f"Loaded {len(df)} rows from ecommerce_churn.csv")

    eng = engine()
    with eng.begin() as conn:
        # Check if already uploaded
        existing = conn.execute(
            text("SELECT COUNT(*) FROM learning_rows WHERE company_id = :cid"),
            {"cid": company_id}
        ).scalar()
        if existing > 0:
            print(f"Already have {existing} rows for {company_id}, skipping insert.")
            return

        # Create batch
        batch_id = conn.execute(text(
            "INSERT INTO upload_batches "
            "(company_id, username, source_file, upload_type, row_count, high_risk_count, accepted_rows, created_at) "
            "VALUES (:cid, 'system', 'ecommerce_churn.csv', 'learning', :n, 0, :n, NOW()) "
            "RETURNING id"
        ), {"cid": company_id, "n": len(df)}).scalar()

        # Insert rows in chunks
        inserted = 0
        for start in range(0, len(df), 200):
            chunk = df.iloc[start:start+200]
            values = []
            for idx, row in chunk.iterrows():
                row_dict = row.to_dict()
                values.append({
                    "bid": batch_id,
                    "cid": company_id,
                    "custid": str(row_dict.get("customerID", f"ROW-{idx}")),
                    "churn": str(row_dict.get("Churn", "No")),
                    "rj": json.dumps(row_dict),
                })
            conn.execute(text(
                "INSERT INTO learning_rows (batch_id, company_id, customer_id, churn, status, row_json, created_at) "
                "VALUES (:bid, :cid, :custid, :churn, 'queued', :rj, NOW())"
            ), values)
            inserted += len(chunk)
            if inserted % 1000 == 0:
                print(f"  inserted {inserted} / {len(df)} ...")

    # Verify
    with eng.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM learning_rows WHERE company_id = :cid"),
            {"cid": company_id}
        ).scalar()
    print(f"Done! {total} rows persisted for {company_id}.")
    print("Now run: python scripts/train_and_save.py all")

if __name__ == "__main__":
    main()
