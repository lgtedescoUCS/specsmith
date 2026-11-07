"""Application settings, read from environment variables or a local .env."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Hosted Postgres providers hand out URLs without a driver (postgres:// or
# postgresql://). SQLAlchemy reads a bare scheme as psycopg2, which is not
# installed; this app uses psycopg 3.
_DRIVERLESS_SCHEMES = ("postgres://", "postgresql://")
_PSYCOPG_SCHEME = "postgresql+psycopg://"


class Settings(BaseSettings):
    # extra="ignore": .env may hold settings for later phases that this
    # class does not declare yet.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # No default on purpose: a missing URL should fail loudly instead of
    # silently connecting to whatever database happens to be on localhost.
    database_url: str

    @field_validator("database_url")
    @classmethod
    def _use_psycopg_driver(cls, url: str) -> str:
        # Only the scheme changes; credentials and query parameters such as
        # sslmode=require are kept as given.
        for scheme in _DRIVERLESS_SCHEMES:
            if url.startswith(scheme):
                return _PSYCOPG_SCHEME + url[len(scheme) :]
        return url

    # Without a connect timeout, a database that silently drops connections
    # makes every request (and /health) hang instead of failing.
    database_connect_timeout_seconds: int = 5

    # Optional so the app, /health and the offline test suite run without a
    # key. Extraction requests fail with 503 until one is set. SecretStr
    # keeps the key out of reprs and logs.
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    llm_timeout_seconds: float = Field(default=30, gt=0)
    prompt_version: str = "v3"
    max_extraction_retries: int = Field(default=2, ge=0)
    # Extractions the API runs in a rolling 24 hours, across all clients,
    # so a public deployment cannot run up an open-ended token bill.
    max_extractions_per_day: int = Field(default=20, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the env
