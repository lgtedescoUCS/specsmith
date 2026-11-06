"""Application settings, read from environment variables or a local .env."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env may hold settings for later phases that this
    # class does not declare yet.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # No default on purpose: a missing URL should fail loudly instead of
    # silently connecting to whatever database happens to be on localhost.
    database_url: str
    # Without a connect timeout, a database that silently drops connections
    # makes every request (and /health) hang instead of failing.
    database_connect_timeout_seconds: int = 5

    # Optional so the app, /health and the offline test suite run without a
    # key. Extraction requests fail with 503 until one is set. SecretStr
    # keeps the key out of reprs and logs.
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    llm_timeout_seconds: float = Field(default=30, gt=0)
    prompt_version: str = "v1"
    max_extraction_retries: int = Field(default=2, ge=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the env
