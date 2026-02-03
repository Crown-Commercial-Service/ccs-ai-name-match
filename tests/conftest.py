"""Pytest configuration and shared fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_candidates():
    """Sample UK public sector organizations for testing."""
    return [
        "Home Office",
        "HMRC",
        "Ministry of Defence",
        "Cabinet Office",
        "Department for Education",
    ]
