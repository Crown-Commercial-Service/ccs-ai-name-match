from functools import lru_cache
from typing import Optional, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, typed configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime switches
    use_mock_llm: bool = True
    mock_similarity_threshold: float = 0.85

    # Optional prompt file path (e.g. prompts/buyer_match_v4.txt)
    prompt_path: str = ""

    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_deployment_name: str = ""
    azure_openai_api_version: str = ""

    # A client-side HTTP batch is still expanded into one Azure OpenAI request per
    # input. Keep this deliberately conservative to avoid an instantaneous burst.
    # This limit is per API process, so normally run one Uvicorn worker unless the
    # value has been divided across workers.
    llm_max_concurrency: int = 1
    llm_min_request_interval_seconds: float = 1.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def missing_azure_vars(settings: Optional[Settings] = None) -> Tuple[str, ...]:
    s = settings or get_settings()
    missing = []
    if not s.azure_openai_endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not s.azure_openai_key:
        missing.append("AZURE_OPENAI_KEY")
    if not s.azure_openai_deployment_name:
        missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not s.azure_openai_api_version:
        missing.append("AZURE_OPENAI_API_VERSION")
    return tuple(missing)
