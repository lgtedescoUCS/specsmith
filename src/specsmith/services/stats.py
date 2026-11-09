"""Aggregates over everything stored: runs, tokens, cost, specs, the cap.

Read-only and LLM-free: /stats costs one round of small GROUP BY queries
and no tokens. Costs come from the list prices in domain/pricing.py as
they are today, applied to the stored token counts, so they are
estimates, and runs of a model with no known price are counted apart.
"""

from pydantic import BaseModel
from sqlalchemy.orm import Session

from specsmith.db import repository
from specsmith.domain.models import Confidence, ConstraintKind
from specsmith.domain.pricing import estimate_cost_usd
from specsmith.services.specs import LIMIT_WINDOW


class ExtractionStats(BaseModel):
    total: int
    succeeded: int
    failed: int
    # None until there is a run to divide by.
    success_rate: float | None
    avg_attempts: float | None
    avg_latency_ms: float | None


class TokenStats(BaseModel):
    input: int
    output: int


class CostStats(BaseModel):
    total_usd: float
    avg_per_extraction_usd: float | None
    # Runs of a model missing from the price table: not in the totals.
    unpriced_extractions: int


class SpecStats(BaseModel):
    total: int
    by_confidence: dict[str, int]


class DailyLimitStats(BaseModel):
    limit: int
    used_last_24h: int
    remaining: int


class Stats(BaseModel):
    extractions: ExtractionStats
    tokens: TokenStats
    estimated_cost: CostStats
    specs: SpecStats
    constraints_by_kind: dict[str, int]
    daily_limit: DailyLimitStats


def compute_stats(session: Session, *, daily_limit: int) -> Stats:
    usage = repository.run_usage_by_model(session)
    total = sum(u.runs for u in usage)
    succeeded = sum(u.succeeded for u in usage)

    cost_usd = 0.0
    priced = 0
    for u in usage:
        cost = estimate_cost_usd(u.model, u.input_tokens, u.output_tokens)
        if cost is not None:
            cost_usd += cost
            priced += u.runs

    by_confidence = _zero_filled(
        [c.value for c in Confidence], repository.spec_counts_by_confidence(session)
    )
    used = repository.recent_extraction_runs(session, window=LIMIT_WINDOW).count

    return Stats(
        extractions=ExtractionStats(
            total=total,
            succeeded=succeeded,
            failed=total - succeeded,
            success_rate=_ratio(succeeded, total, 4),
            avg_attempts=_ratio(sum(u.attempts for u in usage), total, 2),
            avg_latency_ms=_ratio(sum(u.latency_ms for u in usage), total, 0),
        ),
        tokens=TokenStats(
            input=sum(u.input_tokens for u in usage),
            output=sum(u.output_tokens for u in usage),
        ),
        estimated_cost=CostStats(
            total_usd=round(cost_usd, 6),
            avg_per_extraction_usd=_ratio(cost_usd, priced, 6),
            unpriced_extractions=total - priced,
        ),
        specs=SpecStats(total=sum(by_confidence.values()), by_confidence=by_confidence),
        constraints_by_kind=_zero_filled(
            [k.value for k in ConstraintKind],
            repository.constraint_counts_by_kind(session),
        ),
        daily_limit=DailyLimitStats(
            limit=daily_limit,
            used_last_24h=used,
            # The cap is soft under concurrency, so used can pass it.
            remaining=max(0, daily_limit - used),
        ),
    )


def _ratio(part: float, whole: int, digits: int) -> float | None:
    return None if whole == 0 else round(part / whole, digits)


def _zero_filled(known: list[str], counts: dict[str, int]) -> dict[str, int]:
    # Every known key appears, 0 included, so clients need no defaults.
    # A stored value outside the list (none today) is kept, not dropped.
    return {key: counts.get(key, 0) for key in known} | counts
