"""extraction_run persistence against a real PostgreSQL. Skipped without it.

The LLM is always the FakeProvider: no network, no tokens.
"""

import json
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import Connection, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from specsmith.db import repository
from specsmith.db import session as db_session_module
from specsmith.db.models import ExtractionRunRow, RequirementRow, SpecRow
from specsmith.domain.errors import (
    DomainError,
    ExtractionFailedError,
    ExtractionTimeoutError,
    SpecNotFoundError,
)
from specsmith.domain.models import ExtractionMeta, Spec
from specsmith.llm.base import LLMResponse, LLMTimeoutError
from specsmith.llm.fake import FakeProvider, ScriptedItem
from specsmith.llm.prompts import load_prompt
from specsmith.services.extraction import ExtractionService
from specsmith.services.specs import create_spec_from_text

RAW_TEXT = "Client wants 650 plates/day on the tempering line."

SPEC = Spec.model_validate(
    {
        "objective": "Increase tempering line throughput",
        "current_state": {"throughput": "400 plates/day"},
        "target_state": {"throughput": "650 plates/day"},
        "constraints": [{"kind": "deadline", "value": "operational by March"}],
        "open_questions": ["How many shifts per day?"],
        "confidence": "medium",
    }
)

META = ExtractionMeta(
    model="fake-model",
    prompt_version="v1",
    input_tokens=412,
    output_tokens=287,
    latency_ms=2140,
    attempts=2,
)

VALID_DRAFT = json.dumps(
    {
        "objective": "Increase tempering line throughput",
        "current_state": [{"name": "throughput", "value": "400 plates/day"}],
        "target_state": [{"name": "throughput", "value": "650 plates/day"}],
        "constraints": [{"kind": "deadline", "value": "operational by March"}],
        "open_questions": ["How many shifts per day?"],
        "confidence": "medium",
    }
)


def count(conn: Connection | Session, model: type) -> int:
    return conn.scalar(select(func.count()).select_from(model)) or 0


def only_run(session: Session) -> ExtractionRunRow:
    return session.scalars(select(ExtractionRunRow)).one()


def extractor(*responses: ScriptedItem) -> ExtractionService:
    return ExtractionService(FakeProvider(responses), load_prompt("v1"), max_retries=2)


# --- repository ------------------------------------------------------------


def test_success_stores_requirement_spec_and_linked_run(db_session: Session) -> None:
    stored = repository.record_extraction_success(
        db_session, raw_text=RAW_TEXT, source="test", spec=SPEC, meta=META
    )

    run = only_run(db_session)
    assert (run.spec_id, run.requirement_id) == (stored.id, stored.requirement_id)
    assert (run.succeeded, run.error) == (True, None)
    assert (
        run.model,
        run.prompt_version,
        run.input_tokens,
        run.output_tokens,
        run.latency_ms,
        run.attempts,
    ) == ("fake-model", "v1", 412, 287, 2140, 2)
    assert run.created_at is not None
    assert stored == repository.get_spec(db_session, stored.id)


def test_failure_stores_requirement_and_run_without_spec(db_session: Session) -> None:
    run_id = repository.record_extraction_failure(
        db_session, raw_text=RAW_TEXT, source="test", meta=META, error="bad output"
    )

    run = only_run(db_session)
    assert run.id == run_id
    assert (run.succeeded, run.error, run.spec_id) == (False, "bad output", None)
    assert run.requirement.raw_text == RAW_TEXT
    assert count(db_session, SpecRow) == 0


# --- use case --------------------------------------------------------------


def test_create_spec_records_a_successful_run_with_retries(
    db_session: Session,
) -> None:
    stored = create_spec_from_text(
        db_session,
        extractor(
            LLMResponse("not json", "fake-model", input_tokens=10, output_tokens=1),
            LLMResponse(VALID_DRAFT, "fake-model", input_tokens=20, output_tokens=2),
        ),
        raw_text=RAW_TEXT,
        source=None,
        daily_limit=None,
    )

    run = only_run(db_session)
    assert run.spec_id == stored.id
    assert (run.attempts, run.input_tokens, run.output_tokens) == (2, 30, 3)
    assert run.succeeded is True


def test_create_spec_records_a_failed_run_then_raises(db_session: Session) -> None:
    with pytest.raises(ExtractionFailedError) as exc_info:
        create_spec_from_text(
            db_session,
            extractor("bad", "bad", "bad"),
            raw_text=RAW_TEXT,
            source="test",
            daily_limit=None,
        )

    run = only_run(db_session)
    assert (run.succeeded, run.spec_id, run.attempts) == (False, None, 3)
    assert run.error == exc_info.value.reason
    assert "Invalid JSON" in run.error
    assert count(db_session, SpecRow) == 0


def test_create_spec_records_a_timeout(db_session: Session) -> None:
    with pytest.raises(ExtractionTimeoutError):
        create_spec_from_text(
            db_session,
            extractor(LLMTimeoutError("no answer in 30 s")),
            raw_text=RAW_TEXT,
            source="test",
            daily_limit=None,
        )

    run = only_run(db_session)
    assert (run.succeeded, run.attempts, run.input_tokens) == (False, 1, 0)
    assert "no answer in 30 s" in (run.error or "")


# --- get_session keeps failed runs (decision C) ----------------------------


def drive_session(work: Callable[[Session], None], error: BaseException | None) -> None:
    """Run get_session as FastAPI does: work inside, then return or raise."""
    gen = db_session_module.get_session()
    session = next(gen)
    work(session)
    if error is None:
        with pytest.raises(StopIteration):
            next(gen)
    else:
        with pytest.raises(type(error)):
            gen.throw(error)


def add_failed_run(session: Session) -> None:
    repository.record_extraction_failure(
        session, raw_text=RAW_TEXT, source=None, meta=META, error="x"
    )


def test_failed_extraction_error_commits_the_recorded_run(
    connection: Connection,
) -> None:
    drive_session(add_failed_run, ExtractionFailedError("x", META))

    assert count(connection, ExtractionRunRow) == 1


def test_timeout_error_commits_the_recorded_run(connection: Connection) -> None:
    drive_session(add_failed_run, ExtractionTimeoutError("x", META))

    assert count(connection, ExtractionRunRow) == 1


@pytest.mark.parametrize(
    "error",
    [SpecNotFoundError(uuid4()), RuntimeError("boom")],
    ids=["other-domain-error", "unexpected-error"],
)
def test_other_errors_still_roll_back(
    connection: Connection, error: BaseException
) -> None:
    drive_session(add_failed_run, error)

    assert count(connection, ExtractionRunRow) == 0
    assert count(connection, RequirementRow) == 0


def test_normal_return_commits(connection: Connection) -> None:
    drive_session(add_failed_run, None)

    assert count(connection, ExtractionRunRow) == 1


def test_only_extraction_errors_keep_the_transaction() -> None:
    assert DomainError.keeps_transaction is False
    assert SpecNotFoundError.keeps_transaction is False
    assert ExtractionFailedError.keeps_transaction is True
    assert ExtractionTimeoutError.keeps_transaction is True


# --- table constraints -----------------------------------------------------


def run_values(**overrides: Any) -> dict[str, Any]:
    return {
        "model": "m",
        "prompt_version": "v1",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
        "attempts": 1,
        "succeeded": False,
        "error": "x",
    } | overrides


@pytest.mark.parametrize(
    "overrides, constraint",
    [
        ({"succeeded": True, "error": "x"}, "error_set_iff_failed"),
        ({"succeeded": False, "error": None}, "error_set_iff_failed"),
        ({"input_tokens": -1}, "metrics_in_range"),
        ({"latency_ms": -1}, "metrics_in_range"),
        ({"attempts": 0}, "metrics_in_range"),
    ],
)
def test_check_constraints_reject_inconsistent_runs(
    db_session: Session, overrides: dict[str, Any], constraint: str
) -> None:
    db_session.add(
        ExtractionRunRow(
            requirement=RequirementRow(raw_text=RAW_TEXT), **run_values(**overrides)
        )
    )

    with pytest.raises(IntegrityError, match=f"ck_extraction_run_{constraint}"):
        db_session.flush()


def test_a_spec_belongs_to_one_run_at_most(db_session: Session) -> None:
    stored = repository.record_extraction_success(
        db_session, raw_text=RAW_TEXT, source=None, spec=SPEC, meta=META
    )
    db_session.add(
        ExtractionRunRow(
            requirement_id=stored.requirement_id,
            spec_id=stored.id,
            **run_values(succeeded=True, error=None),
        )
    )

    with pytest.raises(IntegrityError, match="uq_extraction_run_spec_id"):
        db_session.flush()


def test_deleting_a_spec_keeps_its_run(db_session: Session) -> None:
    stored = repository.record_extraction_success(
        db_session, raw_text=RAW_TEXT, source=None, spec=SPEC, meta=META
    )

    db_session.execute(delete(SpecRow).where(SpecRow.id == stored.id))
    db_session.expire_all()

    run = only_run(db_session)
    assert (run.spec_id, run.succeeded) == (None, True)


def test_deleting_a_requirement_deletes_its_runs(db_session: Session) -> None:
    add_failed_run(db_session)

    db_session.execute(delete(RequirementRow))

    assert count(db_session, ExtractionRunRow) == 0


def test_meta_rejects_values_that_would_not_fit_the_columns() -> None:
    with pytest.raises(ValidationError, match="model"):
        ExtractionMeta.model_validate(META.model_dump() | {"model": "m" * 81})
    with pytest.raises(ValidationError, match="prompt_version"):
        ExtractionMeta.model_validate(META.model_dump() | {"prompt_version": "v" * 21})
