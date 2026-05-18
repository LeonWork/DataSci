"""
Product storage helpers for ChurnGuard.

This module keeps the pilot app useful without requiring paid infrastructure:
SQLite is the default local store, while the tables mirror the entities we will
move into Postgres/Neon when the first real company uploads arrive.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPANY_ID = "default"
DEFAULT_COMPANY_NAME = "ChurnGuard Pilot"
DEFAULT_DB_PATH = (
    Path("/tmp") / "churnguard.sqlite3"
    if os.getenv("VERCEL")
    else ROOT / "data" / "churnguard.sqlite3"
)


def storage_path() -> Path:
    return Path(os.getenv("CHURNGUARD_DB_PATH", str(DEFAULT_DB_PATH)))


def storage_backend() -> str:
    if os.getenv("DATABASE_URL"):
        return "sqlite pilot store; DATABASE_URL configured for upcoming Postgres migration"
    return "sqlite pilot store"


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _add_column_if_missing(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    if column not in _columns(connection, table):
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


@contextmanager
def _connect():
    path = storage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_storage() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS companies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                plan TEXT NOT NULL DEFAULT 'pilot',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS workspace_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id TEXT NOT NULL,
                username TEXT NOT NULL,
                email TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TEXT,
                UNIQUE(company_id, username),
                FOREIGN KEY (company_id) REFERENCES companies(id)
            );

            CREATE TABLE IF NOT EXISTS prediction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id TEXT NOT NULL,
                username TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies(id)
            );

            CREATE TABLE IF NOT EXISTS upload_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id TEXT NOT NULL,
                username TEXT NOT NULL,
                source_file TEXT NOT NULL,
                upload_type TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                high_risk_count INTEGER NOT NULL DEFAULT 0,
                accepted_rows INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies(id)
            );

            CREATE TABLE IF NOT EXISTS learning_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                company_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                churn TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                row_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (batch_id) REFERENCES upload_batches(id),
                FOREIGN KEY (company_id) REFERENCES companies(id)
            );
            """
        )
        _add_column_if_missing(connection, "companies", "plan", "TEXT NOT NULL DEFAULT 'pilot'")
        _add_column_if_missing(connection, "companies", "status", "TEXT NOT NULL DEFAULT 'active'")
        connection.execute(
            "INSERT OR IGNORE INTO companies (id, name) VALUES (?, ?)",
            (DEFAULT_COMPANY_ID, DEFAULT_COMPANY_NAME),
        )


def ensure_company(company_id: str, name: str | None = None) -> None:
    init_storage()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO companies (id, name)
            VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET name = excluded.name
            """,
            (company_id, name or DEFAULT_COMPANY_NAME),
        )


def upsert_workspace_member(
    *,
    company_id: str,
    username: str,
    email: str = "",
    role: str = "analyst",
    company_name: str | None = None,
) -> None:
    ensure_company(company_id, company_name)
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO workspace_members (company_id, username, email, role, last_seen_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(company_id, username)
            DO UPDATE SET
                email = excluded.email,
                role = excluded.role,
                last_seen_at = CURRENT_TIMESTAMP
            """,
            (company_id, username, email, role),
        )


def workspace_overview(company_id: str) -> dict:
    init_storage()
    with _connect() as connection:
        company = connection.execute(
            "SELECT id, name, plan, status, created_at FROM companies WHERE id = ?",
            (company_id,),
        ).fetchone()
        members = connection.execute(
            """
            SELECT username, email, role, created_at, last_seen_at
            FROM workspace_members
            WHERE company_id = ?
            ORDER BY
                CASE role WHEN 'owner' THEN 0 WHEN 'analyst' THEN 1 ELSE 2 END,
                username
            """,
            (company_id,),
        ).fetchall()

    if company is None:
        return {
            "company_id": company_id,
            "company_name": DEFAULT_COMPANY_NAME,
            "plan": "pilot",
            "status": "active",
            "members": [],
        }

    return {
        "company_id": company["id"],
        "company_name": company["name"],
        "plan": company["plan"],
        "status": company["status"],
        "members": [dict(member) for member in members],
    }


def record_prediction_event(
    *,
    username: str,
    customer_id: str,
    churn_probability: float,
    risk_level: str,
    model_version: str,
    company_id: str = DEFAULT_COMPANY_ID,
) -> None:
    init_storage()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO prediction_events (
                company_id, username, customer_id, churn_probability, risk_level, model_version
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (company_id, username, customer_id, churn_probability, risk_level, model_version),
        )


def record_upload_batch(
    *,
    username: str,
    source_file: str,
    upload_type: str,
    row_count: int,
    high_risk_count: int = 0,
    accepted_rows: int = 0,
    company_id: str = DEFAULT_COMPANY_ID,
) -> int:
    init_storage()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO upload_batches (
                company_id, username, source_file, upload_type, row_count, high_risk_count, accepted_rows
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_id,
                username,
                source_file,
                upload_type,
                row_count,
                high_risk_count,
                accepted_rows,
            ),
        )
        return int(cursor.lastrowid)


def record_learning_rows(
    *,
    batch_id: int,
    rows: Iterable[dict],
    company_id: str = DEFAULT_COMPANY_ID,
) -> None:
    init_storage()
    payload = [
        (
            batch_id,
            company_id,
            str(row.get("customerID") or row.get("customer_id") or "UNKNOWN"),
            str(row.get("Churn")),
            json.dumps(row, default=str, sort_keys=True),
        )
        for row in rows
    ]
    if not payload:
        return

    with _connect() as connection:
        connection.executemany(
            """
            INSERT INTO learning_rows (batch_id, company_id, customer_id, churn, row_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            payload,
        )


def admin_summary(
    *,
    model_type: str,
    model_version: str,
    model_auc: float | None,
    company_id: str = DEFAULT_COMPANY_ID,
) -> dict:
    init_storage()
    with _connect() as connection:
        company = connection.execute(
            "SELECT name FROM companies WHERE id = ?",
            (company_id,),
        ).fetchone()
        totals = connection.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) AS high_risk_predictions
            FROM prediction_events
            WHERE company_id = ?
            """,
            (company_id,),
        ).fetchone()
        uploads = connection.execute(
            """
            SELECT
                COUNT(*) AS csv_upload_batches,
                MAX(created_at) AS latest_upload_at
            FROM upload_batches
            WHERE company_id = ?
            """,
            (company_id,),
        ).fetchone()
        learning = connection.execute(
            """
            SELECT COUNT(*) AS learning_rows_queued
            FROM learning_rows
            WHERE company_id = ? AND status = 'queued'
            """,
            (company_id,),
        ).fetchone()

    queued = int(learning["learning_rows_queued"] or 0)
    return {
        "storage_backend": storage_backend(),
        "company_name": company["name"] if company else DEFAULT_COMPANY_NAME,
        "total_predictions": int(totals["total_predictions"] or 0),
        "high_risk_predictions": int(totals["high_risk_predictions"] or 0),
        "csv_upload_batches": int(uploads["csv_upload_batches"] or 0),
        "learning_rows_queued": queued,
        "latest_upload_at": uploads["latest_upload_at"],
        "model_type": model_type,
        "model_version": model_version,
        "model_auc": model_auc,
        "retrain_recommended": queued >= 100,
    }
