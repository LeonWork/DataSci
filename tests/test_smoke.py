import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Verify that the API liveness endpoint is up."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_dashboard_loading():
    """Verify that the web application dashboard serves successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"ChurnGuard" in response.content

def test_protected_apis_reject_unauthenticated():
    """Verify that protected endpoints reject traffic without a bearer token."""
    endpoints = [
        ("GET", "/admin/summary"),
        ("GET", "/model-info"),
        ("POST", "/predict"),
    ]
    
    for method, path in endpoints:
        response = client.request(method, path)
        assert response.status_code == 401, f"Expected 401 for {method} {path}"

def test_database_connection():
    """Verify that the application can connect to the database engine."""
    from src.api.storage import engine
    from sqlalchemy import text
    
    try:
        with engine().connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar()
            assert result == 1
    except Exception as e:
        pytest.fail(f"Database connection failed: {e}")
