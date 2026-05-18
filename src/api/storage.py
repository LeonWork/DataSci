"""
Product storage helpers for ChurnGuard.

SQLAlchemy lets the same application use free local SQLite during development
and Neon/Postgres in production by setting DATABASE_URL.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    case,
    create_engine,
    func,
    select,
    update,
)
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPANY_ID = "default"
DEFAULT_COMPANY_NAME = "ChurnGuard Pilot"
DEFAULT_DB_PATH = (
    Path("/tmp") / "churnguard.sqlite3"
    if os.getenv("VERCEL")
    else ROOT / "data" / "churnguard.sqlite3"
)

metadata = MetaData()

companies = Table(
    "companies",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("plan", String(40), nullable=False, default="pilot"),
    Column("status", String(40), nullable=False, default="active"),
    Column("created_at", DateTime(timezone=True), nullable=False, default=lambda: _now()),
)

app_users = Table(
    "app_users",
    metadata,
    Column("username", String(80), primary_key=True),
    Column("email", String(255), nullable=False, default=""),
    Column("password_hash", String(255), nullable=False),
    Column("company_id", String(80), ForeignKey("companies.id"), nullable=False),
    Column("role", String(40), nullable=False, default="analyst"),
    Column("created_at", DateTime(timezone=True), nullable=False, default=lambda: _now()),
    Column("last_login_at", DateTime(timezone=True)),
)

workspace_members = Table(
    "workspace_members",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("company_id", String(80), ForeignKey("companies.id"), nullable=False),
    Column("username", String(80), nullable=False),
    Column("email", String(255), nullable=False, default=""),
    Column("role", String(40), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, default=lambda: _now()),
    Column("last_seen_at", DateTime(timezone=True)),
    UniqueConstraint("company_id", "username", name="uq_workspace_member"),
)

prediction_events = Table(
    "prediction_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("company_id", String(80), ForeignKey("companies.id"), nullable=False),
    Column("username", String(80), nullable=False),
    Column("customer_id", String(160), nullable=False),
    Column("churn_probability", Float, nullable=False),
    Column("risk_level", String(40), nullable=False),
    Column("model_version", String(80), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, default=lambda: _now()),
)

upload_batches = Table(
    "upload_batches",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("company_id", String(80), ForeignKey("companies.id"), nullable=False),
    Column("username", String(80), nullable=False),
    Column("source_file", String(255), nullable=False),
    Column("upload_type", String(40), nullable=False),
    Column("row_count", Integer, nullable=False),
    Column("high_risk_count", Integer, nullable=False, default=0),
    Column("accepted_rows", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False, default=lambda: _now()),
)

learning_rows = Table(
    "learning_rows",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("batch_id", Integer, ForeignKey("upload_batches.id"), nullable=False),
    Column("company_id", String(80), ForeignKey("companies.id"), nullable=False),
    Column("customer_id", String(160), nullable=False),
    Column("churn", String(20), nullable=False),
    Column("status", String(60), nullable=False, default="queued"),
    Column("row_json", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, default=lambda: _now()),
)

_ENGINE: Engine | None = None
_ENGINE_KEY: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def database_url() -> str:
    configured = os.getenv("DATABASE_URL")
    if configured:
        if configured.startswith("postgres://"):
            return configured.replace("postgres://", "postgresql+psycopg://", 1)
        if configured.startswith("postgresql://"):
            return configured.replace("postgresql://", "postgresql+psycopg://", 1)
        return configured
    return f"sqlite:///{storage_path()}"


def storage_path() -> Path:
    return Path(os.getenv("CHURNGUARD_DB_PATH", str(DEFAULT_DB_PATH)))


def storage_backend() -> str:
    if database_url().startswith("postgresql+psycopg://"):
        return "postgres"
    return "sqlite local store"


def engine() -> Engine:
    global _ENGINE, _ENGINE_KEY
    key = database_url()
    if _ENGINE is None or _ENGINE_KEY != key:
        if key.startswith("sqlite:///"):
            storage_path().parent.mkdir(parents=True, exist_ok=True)
        _ENGINE = create_engine(key, future=True, pool_pre_ping=True)
        _ENGINE_KEY = key
    return _ENGINE


@contextmanager
def _connect():
    with engine().begin() as connection:
        yield connection


def init_storage() -> None:
    metadata.create_all(engine())
    ensure_company(DEFAULT_COMPANY_ID, DEFAULT_COMPANY_NAME)


def ensure_company(company_id: str, name: str | None = None) -> None:
    metadata.create_all(engine())
    with _connect() as connection:
        existing = connection.execute(
            select(companies.c.id).where(companies.c.id == company_id)
        ).first()
        if existing:
            connection.execute(
                update(companies)
                .where(companies.c.id == company_id)
                .values(name=name or DEFAULT_COMPANY_NAME)
            )
        else:
            connection.execute(
                companies.insert().values(
                    id=company_id,
                    name=name or DEFAULT_COMPANY_NAME,
                    plan="pilot",
                    status="active",
                    created_at=_now(),
                )
            )


def create_db_user(user: dict) -> None:
    init_storage()
    ensure_company(user["company_id"], user.get("company_name") or DEFAULT_COMPANY_NAME)
    with _connect() as connection:
        existing = connection.execute(
            select(app_users.c.username).where(app_users.c.username == user["username"])
        ).first()
        if existing:
            raise ValueError("That username is already taken.")
        email_exists = connection.execute(
            select(app_users.c.username).where(app_users.c.email == user["email"])
        ).first()
        if email_exists:
            raise ValueError("An account with that email already exists.")
        connection.execute(
            app_users.insert().values(
                username=user["username"],
                email=user["email"],
                password_hash=user["password_hash"],
                company_id=user["company_id"],
                role=user["role"],
                created_at=_now(),
            )
        )
    upsert_workspace_member(
        company_id=user["company_id"],
        username=user["username"],
        email=user["email"],
        role=user["role"],
        company_name=user.get("company_name") or DEFAULT_COMPANY_NAME,
    )


def get_db_user(username: str) -> dict | None:
    init_storage()
    with _connect() as connection:
        row = connection.execute(
            select(app_users).where(app_users.c.username == username)
        ).mappings().first()
        if row is None:
            return None
        connection.execute(
            update(app_users)
            .where(app_users.c.username == username)
            .values(last_login_at=_now())
        )
    return dict(row)


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
        existing = connection.execute(
            select(workspace_members.c.id).where(
                workspace_members.c.company_id == company_id,
                workspace_members.c.username == username,
            )
        ).first()
        values = {
            "email": email,
            "role": role,
            "last_seen_at": _now(),
        }
        if existing:
            connection.execute(
                update(workspace_members)
                .where(workspace_members.c.id == existing.id)
                .values(**values)
            )
        else:
            connection.execute(
                workspace_members.insert().values(
                    company_id=company_id,
                    username=username,
                    created_at=_now(),
                    **values,
                )
            )


def workspace_overview(company_id: str) -> dict:
    init_storage()
    with _connect() as connection:
        company = connection.execute(
            select(companies).where(companies.c.id == company_id)
        ).mappings().first()
        members = connection.execute(
            select(
                workspace_members.c.username,
                workspace_members.c.email,
                workspace_members.c.role,
                workspace_members.c.created_at,
                workspace_members.c.last_seen_at,
            )
            .where(workspace_members.c.company_id == company_id)
            .order_by(workspace_members.c.role, workspace_members.c.username)
        ).mappings().all()

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
        "members": [_serialize_member(member) for member in members],
    }


def _serialize_member(member) -> dict:
    data = dict(member)
    for key in ("created_at", "last_seen_at"):
        if isinstance(data.get(key), datetime):
            data[key] = data[key].isoformat()
    return data


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
            prediction_events.insert().values(
                company_id=company_id,
                username=username,
                customer_id=customer_id,
                churn_probability=churn_probability,
                risk_level=risk_level,
                model_version=model_version,
                created_at=_now(),
            )
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
        result = connection.execute(
            upload_batches.insert().values(
                company_id=company_id,
                username=username,
                source_file=source_file,
                upload_type=upload_type,
                row_count=row_count,
                high_risk_count=high_risk_count,
                accepted_rows=accepted_rows,
                created_at=_now(),
            )
        )
        return int(result.inserted_primary_key[0])


def record_learning_rows(
    *,
    batch_id: int,
    rows: Iterable[dict],
    company_id: str = DEFAULT_COMPANY_ID,
) -> None:
    init_storage()
    payload = [
        {
            "batch_id": batch_id,
            "company_id": company_id,
            "customer_id": str(row.get("customerID") or row.get("customer_id") or "UNKNOWN"),
            "churn": str(row.get("Churn")),
            "status": "queued",
            "row_json": json.dumps(row, default=str, sort_keys=True),
            "created_at": _now(),
        }
        for row in rows
    ]
    if not payload:
        return

    with _connect() as connection:
        connection.execute(learning_rows.insert(), payload)


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
            select(companies.c.name).where(companies.c.id == company_id)
        ).first()
        totals = connection.execute(
            select(
                func.count(prediction_events.c.id).label("total_predictions"),
                func.sum(
                    case((prediction_events.c.risk_level == "High", 1), else_=0)
                ).label("high_risk_predictions"),
            ).where(prediction_events.c.company_id == company_id)
        ).mappings().first()
        uploads = connection.execute(
            select(
                func.count(upload_batches.c.id).label("csv_upload_batches"),
                func.max(upload_batches.c.created_at).label("latest_upload_at"),
            ).where(upload_batches.c.company_id == company_id)
        ).mappings().first()
        learning = connection.execute(
            select(func.count(learning_rows.c.id).label("learning_rows_queued")).where(
                learning_rows.c.company_id == company_id,
                learning_rows.c.status == "queued",
            )
        ).mappings().first()

    queued = int(learning["learning_rows_queued"] or 0)
    latest_upload_at = uploads["latest_upload_at"]
    return {
        "storage_backend": storage_backend(),
        "company_name": company.name if company else DEFAULT_COMPANY_NAME,
        "total_predictions": int(totals["total_predictions"] or 0),
        "high_risk_predictions": int(totals["high_risk_predictions"] or 0),
        "csv_upload_batches": int(uploads["csv_upload_batches"] or 0),
        "learning_rows_queued": queued,
        "latest_upload_at": (
            latest_upload_at.isoformat()
            if isinstance(latest_upload_at, datetime)
            else latest_upload_at
        ),
        "model_type": model_type,
        "model_version": model_version,
        "model_auc": model_auc,
        "retrain_recommended": queued >= 100,
    }
