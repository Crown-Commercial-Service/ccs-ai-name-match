from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from langchain_openai import AzureChatOpenAI

from app.config import get_settings, missing_azure_vars


def _build_mock_model(candidates: List[str]):
    # Local import so the repo doesn't break if mock file isn't used in Azure mode
    from app.services.mock_langchain_model import MockChatModelWithCandidates


    settings = get_settings()
    return MockChatModelWithCandidates(
        candidates=candidates,
        similarity_threshold=settings.mock_similarity_threshold,
    )


@lru_cache(maxsize=1)
def _get_azure_model_cached():
    settings = get_settings()
    missing = missing_azure_vars(settings)
    if missing:
        raise RuntimeError("Missing required env vars: " + ", ".join(missing))

    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        openai_api_key=settings.azure_openai_key,
        azure_deployment=settings.azure_openai_deployment_name,
        openai_api_version=settings.azure_openai_api_version,
        temperature=1.0,
    )


def get_chat_model(candidates: Optional[List[str]] = None):
    """
    Returns a chat model that supports: model.invoke(messages)

    - If USE_MOCK_LLM=true (default), returns a mock model that matches against `candidates`.
    - If USE_MOCK_LLM=false, returns AzureChatOpenAI.
    """
    settings = get_settings()

    if settings.use_mock_llm:
        return _build_mock_model(candidates or [])

    return _get_azure_model_cached()
