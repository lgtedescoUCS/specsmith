"""RetryingProvider: backoff around a provider, offline.

The FakeProvider scripts the failures and `sleep` is recorded instead of
waited, so these tests take no time and touch no network.
"""

import logging

import pytest

from specsmith.llm.base import LLMError, LLMTimeoutError, LLMUnavailableError
from specsmith.llm.fake import FakeProvider
from specsmith.llm.retry import RetryingProvider

SCHEMA = {"type": "object"}


def transient(retry_after: float | None = None) -> LLMUnavailableError:
    return LLMUnavailableError(
        "returned 529", retryable=True, retry_after_seconds=retry_after
    )


def retrying(
    *items: object,
    max_retries: int = 2,
    base: float = 1,
    cap: float = 8,
    rand: float = 0.0,
) -> tuple[RetryingProvider, FakeProvider, list[float]]:
    inner = FakeProvider(items)  # type: ignore[arg-type]
    sleeps: list[float] = []
    provider = RetryingProvider(
        inner,
        max_retries=max_retries,
        base_delay_seconds=base,
        max_delay_seconds=cap,
        sleep=sleeps.append,
        rand=lambda: rand,
    )
    return provider, inner, sleeps


def test_success_needs_no_retry() -> None:
    provider, inner, sleeps = retrying("ok")

    assert provider.extract("p", SCHEMA).content == "ok"
    assert (len(inner.calls), sleeps) == (1, [])


def test_transient_failures_are_retried_with_exponential_backoff() -> None:
    provider, inner, sleeps = retrying(transient(), transient(), "ok")

    assert provider.extract("p", SCHEMA).content == "ok"
    assert len(inner.calls) == 3
    assert sleeps == [1, 2]


def test_every_retry_sends_the_same_prompt_and_schema() -> None:
    provider, inner, _ = retrying(transient(), "ok")

    provider.extract("the prompt", SCHEMA)

    assert {(c.prompt, str(c.schema)) for c in inner.calls} == {
        ("the prompt", str(SCHEMA))
    }


def test_gives_up_after_max_retries_with_the_last_error() -> None:
    last = transient()
    provider, inner, sleeps = retrying(transient(), transient(), last)

    with pytest.raises(LLMUnavailableError) as exc_info:
        provider.extract("p", SCHEMA)

    assert exc_info.value is last
    assert len(inner.calls) == 3
    assert len(sleeps) == 2


def test_zero_max_retries_means_a_single_call() -> None:
    provider, inner, sleeps = retrying(transient(), max_retries=0)

    with pytest.raises(LLMUnavailableError):
        provider.extract("p", SCHEMA)

    assert (len(inner.calls), sleeps) == (1, [])


def test_backoff_is_capped() -> None:
    provider, _, sleeps = retrying(
        transient(), transient(), transient(), transient(), "ok",
        max_retries=4, base=1, cap=5,
    )

    provider.extract("p", SCHEMA)

    assert sleeps == [1, 2, 4, 5]


def test_jitter_shortens_the_delay_by_at_most_a_quarter() -> None:
    provider, _, sleeps = retrying(transient(), transient(), "ok", rand=1.0)

    provider.extract("p", SCHEMA)

    assert sleeps == [0.75, 1.5]


def test_retry_after_from_the_provider_is_honoured() -> None:
    provider, _, sleeps = retrying(transient(retry_after=3.5), "ok")

    provider.extract("p", SCHEMA)

    assert sleeps == [3.5]


def test_retry_after_longer_than_the_cap_is_not_waited_out() -> None:
    error = transient(retry_after=60)
    provider, inner, sleeps = retrying(error, "never sent", cap=8)

    with pytest.raises(LLMUnavailableError) as exc_info:
        provider.extract("p", SCHEMA)

    assert exc_info.value is error
    assert (len(inner.calls), sleeps) == (1, [])


@pytest.mark.parametrize(
    "error",
    [
        LLMUnavailableError("returned 401", retryable=False),
        LLMUnavailableError("returned 400: spend limit"),
        LLMTimeoutError("timed out"),
        LLMError("model declined"),
    ],
    ids=["auth", "spend-limit", "timeout", "refusal"],
)
def test_permanent_failures_and_timeouts_are_not_retried(error: LLMError) -> None:
    provider, inner, sleeps = retrying(error, "never sent")

    with pytest.raises(LLMError) as exc_info:
        provider.extract("p", SCHEMA)

    assert exc_info.value is error
    assert (len(inner.calls), sleeps) == (1, [])


def test_reports_the_inner_model() -> None:
    provider, _, _ = retrying()

    assert provider.model == "fake-model"


def test_each_retry_is_logged_as_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    provider, _, _ = retrying(transient(), "ok")

    with caplog.at_level(logging.WARNING, logger="specsmith.llm.retry"):
        provider.extract("p", SCHEMA)

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.event == "llm_retry"  # type: ignore[attr-defined]
    assert record.retry == 1  # type: ignore[attr-defined]
    assert record.delay_seconds == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"base_delay_seconds": 0},
        {"base_delay_seconds": 10, "max_delay_seconds": 5},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    values = {"max_retries": 2, "base_delay_seconds": 1, "max_delay_seconds": 8}
    values |= kwargs

    with pytest.raises(ValueError):
        RetryingProvider(FakeProvider([]), **values)  # type: ignore[arg-type]
