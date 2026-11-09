"""Use case: turn raw requirement text into a stored spec.

Runs the extraction and records the outcome. An extraction_run row is
written on success and on failure; the caller's transaction must keep it
when an ExtractionError propagates (see specsmith.db.session.get_session).
"""

import logging
import math
from datetime import timedelta

from sqlalchemy.orm import Session

from specsmith.db import repository
from specsmith.domain.errors import DailyExtractionLimitError, ExtractionError
from specsmith.domain.models import ExtractionMeta, StoredSpec
from specsmith.services.extraction import ExtractionService

# Rolling window for the daily cap: no time zone decides when "today" ends.
LIMIT_WINDOW = timedelta(days=1)

logger = logging.getLogger(__name__)


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
        run_id = repository.record_extraction_failure(
            session, raw_text=raw_text, source=source, meta=exc.meta, error=exc.reason
        )
        # The reason describes the model's answer or the provider failure,
        # never the client's text; the full record is in extraction_run.
        logger.warning(
            "extraction failed",
            extra=_run_fields(exc.meta)
            | {
                "succeeded": False,
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "reason": exc.reason,
            },
        )
        raise
    stored = repository.record_extraction_success(
        session, raw_text=raw_text, source=source, spec=result.spec, meta=result.meta
    )
    logger.info(
        "extraction succeeded",
        extra=_run_fields(result.meta)
        | {"succeeded": True, "spec_id": stored.id, "confidence": stored.confidence},
    )
    return stored


def _run_fields(meta: ExtractionMeta) -> dict[str, object]:
    # raw_text is never logged: it is client text and lives in the database.
    return {
        "event": "extraction",
        "model": meta.model,
        "prompt_version": meta.prompt_version,
        "attempts": meta.attempts,
        "input_tokens": meta.input_tokens,
        "output_tokens": meta.output_tokens,
        "latency_ms": meta.latency_ms,
    }


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
    retry_after = max(1, math.ceil(frees_in.total_seconds()))
    logger.warning(
        "daily extraction limit reached",
        extra={
            "event": "daily_limit_reached",
            "limit": limit,
            "retry_after_seconds": retry_after,
        },
    )
    raise DailyExtractionLimitError(limit, retry_after_seconds=retry_after)
