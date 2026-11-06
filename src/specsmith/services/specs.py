"""Use case: turn raw requirement text into a stored spec.

Runs the extraction and records the outcome. An extraction_run row is
written on success and on failure; the caller's transaction must keep it
when an ExtractionError propagates (see specsmith.db.session.get_session).
"""

from sqlalchemy.orm import Session

from specsmith.db import repository
from specsmith.domain.errors import ExtractionError
from specsmith.domain.models import StoredSpec
from specsmith.services.extraction import ExtractionService


def create_spec_from_text(
    session: Session,
    extractor: ExtractionService,
    *,
    raw_text: str,
    source: str | None,
) -> StoredSpec:
    """Raises ExtractionFailedError or ExtractionTimeoutError after the
    failed run has been flushed to the session."""
    try:
        result = extractor.extract(raw_text)
    except ExtractionError as exc:
        repository.record_extraction_failure(
            session, raw_text=raw_text, source=source, meta=exc.meta, error=exc.reason
        )
        raise
    return repository.record_extraction_success(
        session, raw_text=raw_text, source=source, spec=result.spec, meta=result.meta
    )
