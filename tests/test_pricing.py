"""Cost estimates from the hardcoded price table. Offline."""

import pytest

from specsmith.domain.models import ExtractionMeta
from specsmith.domain.pricing import estimate_cost, estimate_cost_usd


@pytest.mark.parametrize(
    ("model", "cost"),
    [("claude-sonnet-5-5", 0.012), ("claude-opus-5-5", 0.024), ("other", None)],
)
def test_estimate_cost_usd(model: str, cost: float | None) -> None:
    expected = None if cost is None else pytest.approx(cost)

    assert estimate_cost_usd(model, 1000, 1000) == expected


def test_estimate_cost_of_a_run_uses_its_model_and_tokens() -> None:
    meta = ExtractionMeta(
        model="claude-sonnet-5-5",
        prompt_version="v1",
        input_tokens=1378,
        output_tokens=653,
        latency_ms=1,
        attempts=1,
    )

    assert estimate_cost(meta) == pytest.approx((1378 * 2 + 653 * 10) / 1e6)
