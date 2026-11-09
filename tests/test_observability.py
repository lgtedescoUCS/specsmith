"""JSON log formatter, request id middleware and the uvicorn log config.

Offline: the middleware is exercised on GET /, which needs no database.
"""

import importlib
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from specsmith.main import app
from specsmith.observability import JsonFormatter, request_id_var

LOG_CONFIG = Path(__file__).resolve().parents[1] / "log_config.json"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def no_request_id() -> Iterator[None]:
    token = request_id_var.set(None)
    yield
    request_id_var.reset(token)


def record(msg: str = "hello", **extra: object) -> logging.LogRecord:
    rec = logging.LogRecord("specsmith.test", logging.INFO, __file__, 1, msg, (), None)
    rec.__dict__.update(extra)
    return rec


# --- formatter ---------------------------------------------------------------


def test_formatter_writes_one_json_object(no_request_id: None) -> None:
    line = JsonFormatter().format(record("hello %s", run_id="r1", tokens=12))

    assert "\n" not in line
    entry = json.loads(line)
    assert entry["level"] == "INFO"
    assert entry["logger"] == "specsmith.test"
    assert entry["message"] == "hello %s"
    assert (entry["run_id"], entry["tokens"]) == ("r1", 12)
    assert entry["time"].endswith("+00:00")
    assert "request_id" not in entry


def test_formatter_adds_the_current_request_id() -> None:
    token = request_id_var.set("abc123")
    try:
        entry = json.loads(JsonFormatter().format(record()))
    finally:
        request_id_var.reset(token)

    assert entry["request_id"] == "abc123"


def test_formatter_never_fails_on_odd_values(no_request_id: None) -> None:
    spec_id = UUID(int=1)

    entry = json.loads(JsonFormatter().format(record(spec_id=spec_id)))

    assert entry["spec_id"] == str(spec_id)


def test_formatter_keeps_the_traceback(no_request_id: None) -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        rec = record()
        rec.exc_info = sys.exc_info()

    entry = json.loads(JsonFormatter().format(rec))

    assert "RuntimeError: boom" in entry["exception"]


# --- middleware --------------------------------------------------------------


def test_every_response_carries_a_generated_request_id(client: TestClient) -> None:
    first = client.get("/").headers["x-request-id"]
    second = client.get("/").headers["x-request-id"]

    assert len(first) == 32 and int(first, 16) >= 0
    assert first != second


def test_a_plain_incoming_request_id_is_kept(client: TestClient) -> None:
    response = client.get("/", headers={"X-Request-ID": "gateway-42.a_b"})

    assert response.headers["x-request-id"] == "gateway-42.a_b"


@pytest.mark.parametrize(
    "incoming", ["x" * 65, "has space", 'quote"', "line\\nbreak", ""]
)
def test_an_unsafe_incoming_request_id_is_replaced(
    client: TestClient, incoming: str
) -> None:
    response = client.get("/", headers={"X-Request-ID": incoming})

    assert response.headers["x-request-id"] != incoming
    assert len(response.headers["x-request-id"]) == 32


def test_each_request_is_logged_once_with_its_id(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="specsmith.request"):
        response = client.get("/?q=secret", headers={"X-Request-ID": "req-1"})

    [rec] = [r for r in caplog.records if r.name == "specsmith.request"]
    fields = json.loads(JsonFormatter().format(rec))
    assert fields["event"] == "request"
    assert (fields["method"], fields["path"], fields["status"]) == ("GET", "/", 307)
    assert fields["duration_ms"] >= 0
    # The record is formatted inside the request context in production; here
    # it is checked against the header the client got back.
    assert response.headers["x-request-id"] == "req-1"
    assert "secret" not in json.dumps(fields)


def test_log_calls_made_during_a_request_see_its_id(client: TestClient) -> None:
    seen: list[str | None] = []

    @app.get("/_test_request_id", include_in_schema=False)
    def probe() -> dict[str, str]:
        seen.append(request_id_var.get())
        return {}

    try:
        response = client.get("/_test_request_id", headers={"X-Request-ID": "probe-1"})
    finally:
        app.router.routes.pop()

    assert response.status_code == 200
    assert seen == ["probe-1"]


# --- uvicorn log config -----------------------------------------------------


def test_log_config_routes_everything_through_the_json_formatter() -> None:
    config = json.loads(LOG_CONFIG.read_text(encoding="utf-8"))

    module_name, _, class_name = config["formatters"]["json"]["()"].rpartition(".")
    assert getattr(importlib.import_module(module_name), class_name) is JsonFormatter
    assert config["handlers"]["stdout"]["formatter"] == "json"
    assert config["root"]["handlers"] == ["stdout"]
    # Loggers created before uvicorn applies the config keep working.
    assert config["disable_existing_loggers"] is False
    # The middleware logs each request with its id; uvicorn's own access
    # line would be a second, id-less copy.
    assert config["loggers"]["uvicorn.access"]["propagate"] is False
