"""Tests for prompt file handling."""

import pytest
from pathlib import Path

from app.services.langchain_matcher import _load_prompt_text


class TestPromptLoading:
    """Tests for loading prompt files."""

    def test_load_buyer_match_v1(self):
        """Test loading buyer_match_v1.txt prompt."""
        prompt = _load_prompt_text("prompts/buyer_match_v1.txt")
        
        assert "entity name matching system" in prompt
        assert "{input_name}" in prompt
        assert "{candidates}" in prompt

    def test_load_buyer_match_v2(self):
        """Test loading buyer_match_v2.txt prompt."""
        prompt = _load_prompt_text("prompts/buyer_match_v2.txt")
        
        assert "UK public-sector" in prompt
        assert "DWP" in prompt
        assert "{input_name}" in prompt

    def test_load_buyer_match_v3(self):
        """Test loading buyer_match_v3.txt prompt."""
        prompt = _load_prompt_text("prompts/buyer_match_v3.txt")
        
        assert "UK procurement data" in prompt
        assert "Addenbrookes Hospital" in prompt
        assert "{input_name}" in prompt

    def test_load_buyer_match_v4(self):
        """Test loading buyer_match_v4.txt prompt."""
        prompt = _load_prompt_text("prompts/buyer_match_v4.txt")
        
        assert "cautious but capable" in prompt
        assert "Rutland County Council" in prompt
        assert "{input_name}" in prompt

    def test_nonexistent_prompt_raises_error(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError) as exc:
            _load_prompt_text("prompts/nonexistent.txt")
        
        assert "Prompt file not found" in str(exc.value)
