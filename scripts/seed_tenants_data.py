"""
scripts/seed_tenants_data.py
-----------------------------
Seed training datasets for company-a (Telco schema) and company-b (Ecommerce schema)
into the database to allow them to be trained.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd
from src.api.storage import engine, ensure_company
from sqlalchemy import text

def seed_tenant(company_id: str, company_name: str, csv_path: Path, max_rows: int = 3000) -> None:
    print(f"\n--- Seeding {company_name} ({company_id}) ---")
    ensure_company(company_id, company_name)
    
    if not csv_path.exists():
        print(f"Error: CSV path does not exist: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    # Take a sample or subset to keep training/database sizes reasonable
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    print(f"Ingesting {len(df)} rows from {csv_path.name}...")
    
    eng = engine()
    with eng.begin() as conn:
        # Check if already seeded
        existing = conn.execute(
            text("SELECT COUNT(*) FROM learning_rows WHERE company_id = :cid"),
            {"cid": company_id}
        ).scalar()
        if existing > 0:
            print(f"Tenant {company_id} already has {existing} rows in learning_rows. Skipping.")
            return

        # Insert upload batch
        batch_id = conn.execute(text(
            "INSERT INTO upload_batches "
            "(company_id, username, source_file, upload_type, row_count, high_risk_count, accepted_rows, created_at) "
            "VALUES (:cid, 'system', :file, 'learning', :n, 0, :n, NOW()) "
            "RETURNING id"
        ), {"cid": company_id, "file": csv_path.name, "n": len(df)}).scalar()

        # Insert in chunks of 500 for Postgres performance
        inserted = 0
        chunk_size = 500
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            values = []
            for idx, row in chunk.iterrows():
                row_dict = row.to_dict()
                
                # Normalize target field
                churn_val = str(row_dict.get("Churn", "No"))
                
                # Resolve Customer ID
                cust_id = str(row_dict.get("customerID") or row_dict.get("customer_id") or f"CUST-{idx}")
                
                values.append({
                    "bid": batch_id,
                    "cid": company_id,
                    "custid": cust_id,
                    "churn": churn_val,
                    "rj": json.dumps(row_dict),
                })
            
            conn.execute(text(
                "INSERT INTO learning_rows (batch_id, company_id, customer_id, churn, status, row_json, created_at) "
                "VALUES (:bid, :cid, :custid, :churn, 'queued', :rj, NOW())"
            ), values)
            inserted += len(chunk)
            print(f"  Inserted {inserted} / {len(df)} rows...")
            
    print(f"✓ Successfully seeded {company_id} with {len(df)} rows.")

def main() -> None:
    # 1. Seed company-a with Telco Churn Data
    telco_csv = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    seed_tenant("company-a", "Company A Telco", telco_csv, max_rows=3000)
    
    # 2. Seed company-b with Ecommerce Churn Data
    ecom_csv = ROOT / "data" / "external" / "ecommerce_churn.csv"
    
    # Generate ecommerce data first if it doesn't exist
    if not ecom_csv.exists():
        print("\nGenerating ecommerce_churn.csv first...")
        from scripts.generate_ecommerce_data import main as gen_ecom
        gen_ecom()
        
    seed_tenant("company-b", "Company B E-Commerce", ecom_csv, max_rows=3000)

if __name__ == "__main__":
    main()
