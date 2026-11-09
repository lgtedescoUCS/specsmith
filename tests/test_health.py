from typing import Any

import pytest
from fastapi.testclient import TestClient

from specsmith.db import session as db_session_module
from specsmith.main import app


def test_health_reports_ok_when_database_is_reachable(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_returns_503_when_database_is_unreachable(
    fresh_db_config: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Runs offline: the real engine, pointed at a port where nothing listens.
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none"
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "unreachable"}


def test_engine_sets_connect_timeout_from_settings(
    fresh_db_config: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without it, a database that drops connections silently (as Docker's
    # port forwarding does with the container stopped) hangs /health.
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db.invalid/x")
    monkeypatch.setenv("DATABASE_CONNECT_TIMEOUT_SECONDS", "7")
    captured: dict[str, Any] = {}

    def spy(url: str, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(db_session_module, "create_engine", spy)

    db_session_module.get_engine()

    assert captured["connect_args"] == {"connect_timeout": 7}
