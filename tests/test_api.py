"""HTTP API against a real PostgreSQL. Skipped when it is not running.

The LLM is the FakeProvider injected by conftest (`fake_llm`).
"""

import json
import logging
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, event, func, select, update
from sqlalchemy.orm import Session

from specsmith.db.models import ExtractionRunRow, RequirementRow, SpecRow
from specsmith.domain.errors import LLMNotConfiguredError
from specsmith.llm.base import LLMError, LLMTimeoutError, LLMUnavailableError
from specsmith.llm.fake import FakeProvider
from specsmith.main import (
    RAW_TEXT_MAX_LENGTH,
    app,
    get_daily_extraction_limit,
    get_extractor,
)

RAW_TEXT = "Client wants 650 plates/day on the tempering line, from 400 today."

DRAFT: dict[str, Any] = {
    "objective": "Increase tempering line throughput",
    "current_state": [{"name": "throughput", "value": "400 plates/day"}],
    "target_state": [{"name": "throughput", "value": "650 plates/day"}],
    "constraints": [
        {"kind": "space", "value": "12 m of available floor space"},
        {"kind": "deadline", "value": "operational by March"},
    ],
    "open_questions": ["How many shifts per day?"],
    "confidence": "medium",
}
VALID_ANSWER = json.dumps(DRAFT)

EXPECTED_SPEC = {
    "objective": "Increase tempering line throughput",
    "current_state": {"throughput": "400 plates/day"},
    "target_state": {"throughput": "650 plates/day"},
    "constraints": DRAFT["constraints"],
    "open_questions": ["How many shifts per day?"],
    "confidence": "medium",
}


def payload() -> dict[str, Any]:
    return {"raw_text": RAW_TEXT, "source": "api-test"}


def count(connection: Connection, model: type) -> int:
    return connection.scalar(select(func.count()).select_from(model)) or 0


def only_run(connection: Connection) -> Any:
    return connection.execute(select(ExtractionRunRow)).one()


# --- success ---------------------------------------------------------------


def test_post_extracts_and_returns_201_with_stored_spec(
    client: TestClient, fake_llm: FakeProvider
) -> None:
    fake_llm.queue(VALID_ANSWER)

    response = client.post("/specs", json=payload())

    assert response.status_code == 201
    body = response.json()
    UUID(body["id"])
    UUID(body["requirement_id"])
    assert body["created_at"]
    assert {k: v for k, v in body.items() if k in EXPECTED_SPEC} == EXPECTED_SPEC


def test_post_sends_the_raw_text_to_the_llm(
    client: TestClient, fake_llm: FakeProvider
) -> None:
    fake_llm.queue(VALID_ANSWER)

    client.post("/specs", json=payload())

    assert len(fake_llm.calls) == 1
    assert f"---\n{RAW_TEXT}\n---" in fake_llm.calls[0].prompt


def test_post_records_a_successful_run(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    fake_llm.queue("not json", VALID_ANSWER)

    body = client.post("/specs", json=payload()).json()

    run = only_run(connection)
    assert (run.spec_id, run.succeeded, run.attempts) == (UUID(body["id"]), True, 2)
    assert (run.model, run.prompt_version) == ("fake-model", "v1")


def test_get_spec_returns_what_post_stored(
    client: TestClient, fake_llm: FakeProvider
) -> None:
    fake_llm.queue(VALID_ANSWER)
    created = client.post("/specs", json=payload()).json()

    response = client.get(f"/specs/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_post_without_source_is_accepted(
    client: TestClient, fake_llm: FakeProvider
) -> None:
    fake_llm.queue(VALID_ANSWER)

    assert client.post("/specs", json={"raw_text": RAW_TEXT}).status_code == 201


def test_get_unknown_spec_returns_404(client: TestClient) -> None:
    missing = uuid4()

    response = client.get(f"/specs/{missing}")

    assert response.status_code == 404
    assert response.json() == {"detail": f"Spec {missing} not found"}


def test_get_with_malformed_id_returns_422(client: TestClient) -> None:
    assert client.get("/specs/not-a-uuid").status_code == 422


# --- invalid requests never reach the LLM ----------------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"raw_text": "   "}, id="blank-raw-text"),
        pytest.param({}, id="missing-raw-text"),
        pytest.param(
            {"raw_text": "x" * (RAW_TEXT_MAX_LENGTH + 1)}, id="raw-text-too-long"
        ),
        pytest.param({"raw_text": RAW_TEXT, "source": "x" * 81}, id="source-too-long"),
        pytest.param({"raw_text": RAW_TEXT, "priority": "high"}, id="unknown-field"),
        # The phase 1 shape: a hand-written spec is no longer accepted.
        pytest.param({"raw_text": RAW_TEXT, "spec": EXPECTED_SPEC}, id="spec-field"),
    ],
)
def test_invalid_request_returns_422_without_calling_the_llm(
    client: TestClient,
    fake_llm: FakeProvider,
    connection: Connection,
    body: dict[str, Any],
) -> None:
    response = client.post("/specs", json=body)

    assert response.status_code == 422
    assert fake_llm.calls == []
    assert count(connection, RequirementRow) == 0


# --- extraction failures: explicit status, run kept ------------------------


def test_output_that_never_validates_returns_422_and_keeps_the_run(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    fake_llm.queue("bad", "bad", "bad")

    response = client.post("/specs", json=payload())

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "Could not extract a valid specification from the text"
    assert body["attempts"] == 3
    assert "Invalid JSON" in body["reason"]
    # The failure is on record even though the request failed.
    run = only_run(connection)
    assert (run.succeeded, run.spec_id, run.attempts) == (False, None, 3)
    assert run.error == body["reason"]
    assert count(connection, RequirementRow) == 1
    assert count(connection, SpecRow) == 0


def test_refusal_returns_422_with_the_reason(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    fake_llm.queue(LLMError("Model declined the request (category: cyber)"))

    response = client.post("/specs", json=payload())

    assert response.status_code == 422
    assert "declined" in response.json()["reason"]
    assert count(connection, ExtractionRunRow) == 1


def test_llm_timeout_returns_504_and_keeps_the_run(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    fake_llm.queue(LLMTimeoutError("Anthropic API timed out"))

    response = client.post("/specs", json=payload())

    assert response.status_code == 504
    assert response.json() == {"detail": "The LLM did not answer in time", "attempts": 1}
    run = only_run(connection)
    assert run.succeeded is False
    assert "timed out" in run.error


def test_llm_unavailable_returns_502_and_keeps_the_run(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    fake_llm.queue(LLMUnavailableError("Anthropic API returned 401: invalid key"))

    response = client.post("/specs", json=payload())

    assert response.status_code == 502
    # Upstream details (here, an auth failure) stay out of the response.
    assert response.json() == {"detail": "The LLM provider is unavailable", "attempts": 1}
    assert "401" in only_run(connection).error


def test_missing_llm_configuration_returns_503_and_stores_nothing(
    client: TestClient, connection: Connection
) -> None:
    def not_configured() -> None:
        raise LLMNotConfiguredError()

    app.dependency_overrides[get_extractor] = not_configured

    response = client.post("/specs", json=payload())

    assert response.status_code == 503
    assert response.json() == {"detail": "LLM provider is not configured"}
    assert count(connection, RequirementRow) == 0


def test_failed_commit_is_an_error_not_a_201(
    connection: Connection, fake_llm: FakeProvider
) -> None:
    # Guards the scope="function" choice on the session dependency: the
    # commit must run before the response is sent, so its failure is visible.
    # before_commit fires only for the real commit, not for the internal
    # subtransaction of each flush, so the endpoint itself still succeeds.
    def failing_commit(session: Session) -> None:
        raise RuntimeError("simulated commit failure")

    fake_llm.queue(VALID_ANSWER)
    event.listen(Session, "before_commit", failing_commit)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/specs", json=payload())
    finally:
        event.remove(Session, "before_commit", failing_commit)

    assert response.status_code == 500
    assert count(connection, RequirementRow) == 0


# --- daily extraction cap ----------------------------------------------------


def limit_to(n: int) -> None:
    app.dependency_overrides[get_daily_extraction_limit] = lambda: n


def test_post_over_the_daily_limit_returns_429_without_calling_the_llm(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    limit_to(2)
    fake_llm.queue(VALID_ANSWER, VALID_ANSWER)
    assert client.post("/specs", json=payload()).status_code == 201
    assert client.post("/specs", json=payload()).status_code == 201
    calls_before = len(fake_llm.calls)

    response = client.post("/specs", json=payload())

    assert response.status_code == 429
    assert response.json() == {"detail": "Daily limit of 2 extractions reached"}
    # The test runs in one transaction, so now() is the same for every row:
    # the oldest run frees its slot exactly 24 hours later.
    assert response.headers["Retry-After"] == "86400"
    assert len(fake_llm.calls) == calls_before
    assert count(connection, ExtractionRunRow) == 2
    assert count(connection, RequirementRow) == 2


def test_failed_extractions_count_toward_the_daily_limit(
    client: TestClient, fake_llm: FakeProvider
) -> None:
    limit_to(1)
    fake_llm.queue("bad", "bad", "bad")
    assert client.post("/specs", json=payload()).status_code == 422

    response = client.post("/specs", json=payload())

    assert response.status_code == 429


def test_runs_older_than_a_day_free_their_slot(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    limit_to(1)
    fake_llm.queue(VALID_ANSWER, VALID_ANSWER)
    assert client.post("/specs", json=payload()).status_code == 201
    connection.execute(
        update(ExtractionRunRow).values(
            created_at=func.now() - timedelta(hours=24, seconds=1)
        )
    )

    response = client.post("/specs", json=payload())

    assert response.status_code == 201


def test_retry_after_counts_down_from_the_oldest_run_in_the_window(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    limit_to(1)
    fake_llm.queue(VALID_ANSWER)
    assert client.post("/specs", json=payload()).status_code == 201
    connection.execute(
        update(ExtractionRunRow).values(created_at=func.now() - timedelta(hours=23))
    )

    response = client.post("/specs", json=payload())

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "3600"


# --- logging ---------------------------------------------------------------


def extraction_records(caplog: pytest.LogCaptureFixture) -> list[Any]:
    return [r for r in caplog.records if getattr(r, "event", None) == "extraction"]


def test_successful_extraction_is_logged_without_the_raw_text(
    client: TestClient, fake_llm: FakeProvider, caplog: pytest.LogCaptureFixture
) -> None:
    fake_llm.queue(VALID_ANSWER)

    with caplog.at_level(logging.INFO, logger="specsmith"):
        body = client.post("/specs", json=payload()).json()

    [rec] = extraction_records(caplog)
    assert rec.levelno == logging.INFO
    assert (rec.succeeded, rec.spec_id, rec.attempts) == (True, UUID(body["id"]), 1)
    assert (rec.model, rec.prompt_version) == ("fake-model", "v1")
    assert (rec.input_tokens, rec.output_tokens) == (100, 50)
    assert all(RAW_TEXT not in str(value) for value in vars(rec).values())


def test_failed_extraction_is_logged_with_its_run_id(
    client: TestClient,
    fake_llm: FakeProvider,
    connection: Connection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_llm.queue(LLMTimeoutError("took too long"))

    with caplog.at_level(logging.INFO, logger="specsmith"):
        client.post("/specs", json=payload())

    [rec] = extraction_records(caplog)
    assert rec.levelno == logging.WARNING
    assert (rec.succeeded, rec.error_type) == (False, "ExtractionTimeoutError")
    assert rec.run_id == only_run(connection).id
    assert all(RAW_TEXT not in str(value) for value in vars(rec).values())
