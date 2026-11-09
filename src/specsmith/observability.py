"""Structured JSON logs and a correlation id per request.

Stdlib only, and nothing here runs at import time: uvicorn loads
JsonFormatter from log_config.json before it imports the app, and the test
suite never installs it, so caplog sees plain records.

Every log line written while a request is being served carries that
request's id, which is also returned to the client in X-Request-ID. A user
reporting an error can quote the id, and it finds every line about it.
"""

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "x-request-id"

# A client may send its own id (a gateway, a test script) to correlate its
# logs with ours. Accepted only if short and plain, so a crafted header
# cannot forge log lines or bloat every record; otherwise a fresh one is
# generated.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

# Set at the start of every request, so a value left from an earlier
# request is always replaced before anything logs. uvicorn also runs each
# request in a task of its own, with its own copy of the context.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

request_logger = logging.getLogger("specsmith.request")

# Attributes every LogRecord has, plus uvicorn's color_message (the same
# message with terminal colour codes). Anything else on a record came from
# `extra=` and is written as a field of its own.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "color_message"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, request_id
    (inside a request) and any `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id is not None:
            entry["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in entry:
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        # default=str: an odd extra value (a UUID, a datetime) must never
        # make a log call fail.
        return json.dumps(entry, default=str, ensure_ascii=False)


def _incoming_request_id(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name == REQUEST_ID_HEADER.encode():
            candidate = value.decode("latin-1")
            if _VALID_REQUEST_ID.fullmatch(candidate):
                return candidate
    return None


class RequestContextMiddleware:
    """Assigns the request id, returns it in X-Request-ID and logs one line
    per request (method, path, status, duration).

    A pure ASGI middleware, not @app.middleware("http"): it adds the header
    to every response, including those built by exception handlers, without
    re-wrapping the body, and the id it sets in the context is visible to
    everything the request runs.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex
        request_id_var.set(request_id)
        status_code = 500
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            # Path only, not the query string, and never the body: requests
            # carry client requirement text, which stays in the database.
            request_logger.info(
                "request",
                extra={
                    "event": "request",
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                },
            )
