"""Application settings, read from environment variables or a local .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env also holds settings for later phases (LLM keys,
    # prompt version) that this class does not declare yet.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # No default on purpose: a missing URL should fail loudly instead of
    # silently connecting to whatever database happens to be on localhost.
    database_url: str
    # Without a connect timeout, a database that silently drops connections
    # makes every request (and /health) hang instead of failing.
    database_connect_timeout_seconds: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the env
