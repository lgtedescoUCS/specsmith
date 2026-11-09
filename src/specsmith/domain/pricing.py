"""Estimated cost of LLM usage, from hardcoded list prices.

Shared by the evaluation reports and the /stats endpoint. Prices drift
when the provider changes them, so every number derived from here is an
estimate, and a model missing from the table gets no estimate at all
rather than a wrong one.
"""

from specsmith.domain.models import ExtractionMeta

# USD per million tokens (input, output), standard Messages API requests.
# Checked on 2026-11-06 at
# https://platform.claude.com/docs/en/about-claude/pricing
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int
) -> float | None:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1e6


def estimate_cost(meta: ExtractionMeta) -> float | None:
    return estimate_cost_usd(meta.model, meta.input_tokens, meta.output_tokens)
