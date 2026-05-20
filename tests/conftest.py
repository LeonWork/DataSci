from __future__ import annotations

import os

import pytest


os.environ["CHURNGUARD_TESTING"] = "1"
os.environ.pop("DATABASE_URL", None)


@pytest.fixture(autouse=True)
def _isolated_local_database(tmp_path, monkeypatch):
    monkeypatch.setenv("CHURNGUARD_TESTING", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("CHURNGUARD_DB_PATH", str(tmp_path / "churnguard-test.sqlite3"))

    from src.api.storage import reset_engine

    reset_engine()
    yield
    reset_engine()
