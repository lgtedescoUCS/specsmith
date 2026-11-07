"""Settings parsing that does not involve the LLM. Offline."""

import pytest

from specsmith.config import Settings


def settings(database_url: str) -> Settings:
    return Settings(_env_file=None, database_url=database_url)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "given",
    [
        "postgres://u:p@db.example.test:5432/app?sslmode=require",
        "postgresql://u:p@db.example.test:5432/app?sslmode=require",
    ],
)
def test_driverless_database_url_gets_the_psycopg_driver(given: str) -> None:
    assert (
        settings(given).database_url
        == "postgresql+psycopg://u:p@db.example.test:5432/app?sslmode=require"
    )


def test_database_url_with_a_driver_is_kept_as_is() -> None:
    url = "postgresql+psycopg://specsmith:specsmith@127.0.0.1:5433/specsmith"

    assert settings(url).database_url == url


def test_only_the_scheme_is_rewritten() -> None:
    # A password that happens to contain the scheme text stays untouched.
    url = "postgresql://u:postgresql://x@db.example.test/app"

    assert (
        settings(url).database_url
        == "postgresql+psycopg://u:postgresql://x@db.example.test/app"
    )
