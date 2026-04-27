
import pytest
from starlette.testclient import TestClient
from agent import app

def test_health_check_endpoint_returns_ok():
    """Validates that the /health endpoint returns a 200 status and the expected status payload."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"