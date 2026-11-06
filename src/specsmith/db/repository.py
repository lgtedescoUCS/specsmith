"""Data access for specs. Takes and returns domain models, never ORM rows.

Functions here flush but never commit: the caller owns the transaction
(see specsmith.db.session.get_session).
"""

from uuid import UUID

from sqlalchemy.orm import Session

from specsmith.db.models import (
    ConstraintItemRow,
    ExtractionRunRow,
    OpenQuestionRow,
    RequirementRow,
    SpecRow,
)
from specsmith.domain.errors import SpecNotFoundError
from specsmith.domain.models import ExtractionMeta, Spec, StoredSpec


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
