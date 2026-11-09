"""Data access for specs. Takes and returns domain models, never ORM rows.

Functions here flush but never commit: the caller owns the transaction
(see specsmith.db.session.get_session).
"""

from datetime import datetime, timedelta
from typing import NamedTuple
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from specsmith.db.models import (
    ConstraintItemRow,
    ExtractionRunRow,
    OpenQuestionRow,
    RequirementRow,
    SpecRow,
)
from specsmith.domain.errors import RunNotFoundError, SpecNotFoundError
from specsmith.domain.models import (
    Confidence,
    ExtractionMeta,
    Spec,
    StoredRun,
    StoredSpec,
)


def save_spec(
    session: Session, *, raw_text: str, source: str | None, spec: Spec
) -> StoredSpec:
    row = _spec_row(RequirementRow(raw_text=raw_text, source=source), spec)
    session.add(row)
    # Flush, not commit: sends the INSERTs so ids and created_at exist,
    # while leaving commit or rollback to whoever opened the transaction.
    session.flush()
    return _to_domain(row)


def record_extraction_success(
    session: Session,
    *,
    raw_text: str,
    source: str | None,
    spec: Spec,
    meta: ExtractionMeta,
) -> StoredSpec:
    """Requirement, spec and the run that produced it, in one flush."""
    requirement = RequirementRow(raw_text=raw_text, source=source)
    spec_row = _spec_row(requirement, spec)
    session.add(_run_row(requirement, meta, spec=spec_row, error=None))
    session.flush()
    return _to_domain(spec_row)


def record_extraction_failure(
    session: Session,
    *,
    raw_text: str,
    source: str | None,
    meta: ExtractionMeta,
    error: str,
) -> UUID:
    """Requirement and a failed run with no spec. Returns the run id.

    The raw text is kept: without it a failure could not be reproduced.
    """
    requirement = RequirementRow(raw_text=raw_text, source=source)
    run = _run_row(requirement, meta, spec=None, error=error)
    session.add(run)
    session.flush()
    return run.id


class RecentRuns(NamedTuple):
    count: int
    oldest: datetime | None
    now: datetime


def recent_extraction_runs(session: Session, *, window: timedelta) -> RecentRuns:
    """Runs, successful or not, recorded within `window` of the database's
    clock. `now` comes from the database too, so callers compare times on a
    single clock, the one that filled created_at."""
    count, oldest, now = session.execute(
        select(
            func.count(),
            func.min(ExtractionRunRow.created_at),
            func.now(),
        ).where(ExtractionRunRow.created_at > func.now() - window)
    ).one()
    return RecentRuns(count=count, oldest=oldest, now=now)


class ModelUsage(NamedTuple):
    """Totals over every run, successful or not, served by one model."""

    model: str
    runs: int
    succeeded: int
    attempts: int
    input_tokens: int
    output_tokens: int
    latency_ms: int


def run_usage_by_model(session: Session) -> list[ModelUsage]:
    # Grouped by model because cost depends on it: prices are per model,
    # and applying them is the caller's business, not the database's.
    run = ExtractionRunRow
    rows = session.execute(
        select(
            run.model,
            func.count(),
            func.count().filter(run.succeeded),
            func.sum(run.attempts),
            func.sum(run.input_tokens),
            func.sum(run.output_tokens),
            func.sum(run.latency_ms),
        )
        .group_by(run.model)
        .order_by(run.model)
    ).all()
    return [ModelUsage(*row) for row in rows]


def spec_counts_by_confidence(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(SpecRow.confidence, func.count()).group_by(SpecRow.confidence)
    ).all()
    return {confidence: n for confidence, n in rows}


def constraint_counts_by_kind(session: Session) -> dict[str, int]:
    # The aggregate that normalised constraint rows make a plain GROUP BY.
    rows = session.execute(
        select(ConstraintItemRow.kind, func.count()).group_by(ConstraintItemRow.kind)
    ).all()
    return {kind: n for kind, n in rows}


def _run_row(
    requirement: RequirementRow,
    meta: ExtractionMeta,
    *,
    spec: SpecRow | None,
    error: str | None,
) -> ExtractionRunRow:
    return ExtractionRunRow(
        requirement=requirement,
        spec=spec,
        model=meta.model,
        prompt_version=meta.prompt_version,
        input_tokens=meta.input_tokens,
        output_tokens=meta.output_tokens,
        latency_ms=meta.latency_ms,
        attempts=meta.attempts,
        succeeded=error is None,
        error=error,
    )


def _spec_row(requirement: RequirementRow, spec: Spec) -> SpecRow:
    return SpecRow(
        requirement=requirement,
        objective=spec.objective,
        current_state=spec.current_state,
        target_state=spec.target_state,
        confidence=spec.confidence.value,
        constraints=[
            ConstraintItemRow(position=i, kind=c.kind.value, value=c.value)
            for i, c in enumerate(spec.constraints)
        ],
        open_questions=[
            OpenQuestionRow(position=i, text=q)
            for i, q in enumerate(spec.open_questions)
        ],
    )


def get_spec(session: Session, spec_id: UUID) -> StoredSpec:
    row = session.get(SpecRow, spec_id)
    if row is None:
        raise SpecNotFoundError(spec_id)
    return _to_domain(row)


class SpecPage(NamedTuple):
    items: list[StoredSpec]
    # Matching specs across all pages, for the client to know when to stop.
    total: int


def list_specs(
    session: Session,
    *,
    confidence: Confidence | None,
    limit: int,
    offset: int,
) -> SpecPage:
    """Newest first. id breaks ties in created_at (rows inserted in one
    transaction share now()), so pages never overlap or skip a spec."""
    filters = [] if confidence is None else [SpecRow.confidence == confidence.value]
    total = session.scalar(select(func.count()).select_from(SpecRow).where(*filters))
    rows = session.scalars(
        select(SpecRow)
        .where(*filters)
        .order_by(SpecRow.created_at.desc(), SpecRow.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return SpecPage(items=[_to_domain(row) for row in rows], total=total or 0)


def get_run_for_spec(session: Session, spec_id: UUID) -> StoredRun:
    """The run that produced the spec. Raises SpecNotFoundError when there
    is no such spec, RunNotFoundError when the spec has no run."""
    run = session.scalars(
        select(ExtractionRunRow).where(ExtractionRunRow.spec_id == spec_id)
    ).one_or_none()
    if run is not None:
        return _run_to_domain(run)
    if session.get(SpecRow, spec_id) is None:
        raise SpecNotFoundError(spec_id)
    raise RunNotFoundError(spec_id)


def _run_to_domain(row: ExtractionRunRow) -> StoredRun:
    return StoredRun(
        id=row.id,
        requirement_id=row.requirement_id,
        spec_id=row.spec_id,
        succeeded=row.succeeded,
        created_at=row.created_at,
        model=row.model,
        prompt_version=row.prompt_version,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        latency_ms=row.latency_ms,
        attempts=row.attempts,
    )


def _to_domain(row: SpecRow) -> StoredSpec:
    return StoredSpec.model_validate(
        {
            "id": row.id,
            "requirement_id": row.requirement_id,
            "created_at": row.created_at,
            "objective": row.objective,
            "current_state": row.current_state,
            "target_state": row.target_state,
            "confidence": row.confidence,
            "constraints": [
                {"kind": c.kind, "value": c.value} for c in row.constraints
            ],
            "open_questions": [q.text for q in row.open_questions],
        }
    )
