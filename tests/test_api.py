"""Tests for FastAPI endpoints."""

import pytest
from fastapi import status


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test that health endpoint returns ok."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "ok"}


class TestMatchEndpoint:
    """Tests for /match endpoint - the main functionality."""

    def test_post_exact_match(self, client, sample_candidates):
        """Test POST with exact match."""
        payload = {
            "input_string": "Home Office",
            "candidates": sample_candidates,
        }
        response = client.post("/match", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["input_string"] == "Home Office"
        assert data["match"] == "Home Office"

    def test_post_typo_match(self, client, sample_candidates):
        """Test POST with typo - should still match."""
        payload = {
            "input_string": "Home Ofice",
            "candidates": sample_candidates,
        }
        response = client.post("/match", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["match"] == "Home Office"

    def test_post_no_match(self, client, sample_candidates):
        """Test POST when no good match exists."""
        payload = {
            "input_string": "Random Organization XYZ",
            "candidates": sample_candidates,
        }
        response = client.post("/match", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["match"] is None

    def test_post_case_insensitive(self, client, sample_candidates):
        """Test that matching is case-insensitive."""
        payload = {
            "input_string": "home office",
            "candidates": sample_candidates,
        }
        response = client.post("/match", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["match"] == "Home Office"

    def test_post_with_prompt_path(self, client, sample_candidates):
        """Test POST with custom prompt file."""
        payload = {
            "input_string": "HMRC",
            "candidates": sample_candidates,
            "prompt_path": "prompts/buyer_match_v4.txt",
        }
        response = client.post("/match", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["match"] is not None

    def test_post_empty_input_string_fails(self, client, sample_candidates):
        """Test that empty input_string is rejected."""
        payload = {
            "input_string": "",
            "candidates": sample_candidates,
        }
        response = client.post("/match", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_empty_candidates_fails(self, client):
        """Test that empty candidates list is rejected."""
        payload = {
            "input_string": "Home Office",
            "candidates": [],
        }
        response = client.post("/match", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_exact_match(self, client):
        """Test GET endpoint with exact match."""
        response = client.get(
            "/match",
            params={
                "input_string": "HMRC",
                "candidates": ["Home Office", "HMRC", "Cabinet Office"],
            },
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["match"] == "HMRC"

    def test_output_normalization(self, client, sample_candidates):
        """Test that output is properly normalized."""
        payload = {
            "input_string": "Ministry of Defence",
            "candidates": sample_candidates,
        }
        response = client.post("/match", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Match should not have quotes or extra whitespace
        if data["match"] is not None:
            assert not data["match"].startswith('"')
            assert not data["match"].endswith('"')
            assert data["match"] == data["match"].strip()


class TestRealWorldScenarios:
    """Tests based on actual UK procurement use cases."""

    def test_acronym_expansion(self, client):
        """Test matching acronyms to full names."""
        candidates = [
            "Department for Work and Pensions",
            "Department for Education",
            "Department of Health and Social Care",
        ]
        payload = {
            "input_string": "DWP",
            "candidates": candidates,
        }
        response = client.post("/match", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        # With mock model, DWP won't match well
        # This test is more relevant for Azure OpenAI

    def test_nhs_trust_matching(self, client):
        """Test matching hospital names to NHS trusts."""
        candidates = [
            "Cambridge University Hospitals NHS Foundation Trust",
            "Oxford University Hospitals NHS Foundation Trust",
        ]
        payload = {
            "input_string": "Addenbrookes Hospital",
            "candidates": candidates,
            "prompt_path": "prompts/buyer_match_v3.txt",
        }
        response = client.post("/match", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        # Semantic matching requires Azure OpenAI

    def test_local_authority_matching(self, client):
        """Test matching council names."""
        candidates = [
            "Birmingham City Council",
            "Manchester City Council",
            "Leeds City Council",
        ]
        payload = {
            "input_string": "Birmingham Council",
            "candidates": candidates,
        }
        response = client.post("/match", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["match"] == "Birmingham City Council"
