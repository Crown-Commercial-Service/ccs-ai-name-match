"""Tests for mock model behavior."""

import pytest
from unittest.mock import Mock

from app.services.mock_langchain_model import MockChatModelWithCandidates


class TestMockModel:
    """Basic tests for mock model matching logic."""

    def test_exact_match(self):
        """Test exact match returns correct candidate."""
        candidates = ["Home Office", "HMRC", "Cabinet Office"]
        model = MockChatModelWithCandidates(candidates=candidates)
        
        message = Mock()
        message.content = "Home Office"
        
        response = model.invoke([message])
        assert response.content == "Home Office"

    def test_case_insensitive(self):
        """Test matching ignores case."""
        candidates = ["Home Office", "HMRC"]
        model = MockChatModelWithCandidates(candidates=candidates)
        
        message = Mock()
        message.content = "home office"
        
        response = model.invoke([message])
        assert response.content == "Home Office"

    def test_typo_handling(self):
        """Test that typos are handled within threshold."""
        candidates = ["Home Office", "HMRC"]
        model = MockChatModelWithCandidates(candidates=candidates, similarity_threshold=0.85)
        
        message = Mock()
        message.content = "Home Ofice"
        
        response = model.invoke([message])
        assert response.content == "Home Office"

    def test_no_match_returns_none(self):
        """Test that poor matches return None."""
        candidates = ["Home Office", "HMRC"]
        model = MockChatModelWithCandidates(candidates=candidates, similarity_threshold=0.85)
        
        message = Mock()
        message.content = "Completely Different Organization"
        
        response = model.invoke([message])
        assert response.content == "None"

    def test_empty_candidates(self):
        """Test behavior with no candidates."""
        model = MockChatModelWithCandidates(candidates=[])
        
        message = Mock()
        message.content = "Home Office"
        
        response = model.invoke([message])
        assert response.content == "None"

    def test_best_match_selection(self):
        """Test that best matching candidate is selected."""
        candidates = [
            "Ministry of Defence",
            "Ministry of Justice",
            "Home Office",
        ]
        model = MockChatModelWithCandidates(candidates=candidates)
        
        message = Mock()
        message.content = "Ministry of Defense"
        
        response = model.invoke([message])
        assert response.content == "Ministry of Defence"
