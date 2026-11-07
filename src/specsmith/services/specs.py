"""Use case: turn raw requirement text into a stored spec.

Runs the extraction and records the outcome. An extraction_run row is
written on success and on failure; the caller's transaction must keep it
when an ExtractionError propagates (see specsmith.db.session.get_session).
"""

import math
from datetime import timedelta

from sqlalchemy.orm import Session

from specsmith.db import repository
from specsmith.domain.errors import DailyExtractionLimitError, ExtractionError
from specsmith.domain.models import StoredSpec
from specsmith.services.extraction import ExtractionService

# Rolling window for the daily cap: no time zone decides when "today" ends.
LIMIT_WINDOW = timedelta(days=1)


def create_spec_from_text(
    session: Session,
    extractor: ExtractionService,
    *,
    raw_text: str,
    source: str | None,
    daily_limit: int | None,
) -> StoredSpec:
    """Raises DailyExtractionLimitError before calling the LLM when the
    server already ran `daily_limit` extractions in the last 24 hours
    (None means no cap). Raises an ExtractionError after the failed run
    has been flushed to the session."""
    if daily_limit is not None:
        _check_daily_limit(session, daily_limit)
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


def _check_daily_limit(session: Session, limit: int) -> None:
    # Failed runs count too: they spent tokens as well. The cap is soft:
    # requests that arrive together can each see one slot left and all run,
    # so it can be passed by the number of concurrent requests. It protects
    # a demo budget; the hard ceiling is the API key's spend limit.
    recent = repository.recent_extraction_runs(session, window=LIMIT_WINDOW)
    if recent.count < limit or recent.oldest is None:
        return
    # A slot frees when the oldest run in the window ages out of it.
    frees_in = recent.oldest + LIMIT_WINDOW - recent.now
    raise DailyExtractionLimitError(
        limit, retry_after_seconds=max(1, math.ceil(frees_in.total_seconds()))
    )
