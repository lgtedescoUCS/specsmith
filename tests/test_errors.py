"""Errors no domain handler covers: database down and unexpected bugs.

Offline: the database tests point the real engine at a closed port, and
the bug is raised by a throwaway route.
"""

import logging
from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from specsmith.main import app

UNREACHABLE = "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none"


@pytest.fixture
def database_down(fresh_db_config: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", UNREACHABLE)


def test_unreachable_database_returns_503_not_500(
    database_down: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="specsmith.main"):
        response = TestClient(app).get(f"/specs/{uuid4()}")

    assert response.status_code == 503
    # No host, port or driver message in the body.
    assert response.json() == {"detail": "Database unavailable"}
    assert "x-request-id" in response.headers
    [record] = [r for r in caplog.records if r.name == "specsmith.main"]
    assert record.event == "database_unavailable"  # type: ignore[attr-defined]
    assert record.exc_info is not None


@pytest.fixture
def failing_route() -> Iterator[str]:
    path = "/_test_unhandled"

    @app.get(path, include_in_schema=False)
    def boom() -> None:
        raise RuntimeError("bug with a secret detail")

    yield path
    app.router.routes.pop()


def test_unexpected_error_returns_json_500_with_the_request_id(
    failing_route: str,
) -> None:
    response = TestClient(app).get(failing_route, headers={"X-Request-ID": "bug-1"})

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-request-id"] == "bug-1"
    assert response.json() == {"detail": "Internal server error", "request_id": "bug-1"}
    assert "secret" not in response.text


def test_unexpected_error_is_logged_with_its_traceback(
    failing_route: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="specsmith.request"):
        TestClient(app).get(failing_route)

    error, access = [r for r in caplog.records if r.name == "specsmith.request"]
    assert error.levelno == logging.ERROR
    assert error.event == "unhandled_error"  # type: ignore[attr-defined]
    assert error.exc_info is not None
    assert "bug with a secret detail" in str(error.exc_info[1])
    assert access.status == 500  # type: ignore[attr-defined]
