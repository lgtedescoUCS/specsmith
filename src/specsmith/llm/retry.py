"""Exponential backoff around any LLMProvider, for transient failures only.

Vendor-neutral: it wraps the LLMProvider protocol and decides from the
error alone (LLMUnavailableError.retryable and .retry_after_seconds), so it
is tested offline with the FakeProvider and works for any provider.

What is retried and what is not:
- Retryable upstream failures (overload, rate limit, dropped connection):
  retried up to `max_retries` times, waiting base * 2**n seconds, capped at
  `max_delay_seconds`, with jitter so clients that failed together do not
  retry together.
- A wait the provider asked for (retry-after) is honoured when it fits under
  the cap. A longer one is not waited out: the request fails at once rather
  than hold the client for longer than the cap allows.
- Timeouts are not retried. Each one already cost the full timeout, and a
  slow answer would likely be slow again; retrying would multiply the time
  the client waits before its 504.
- Everything else (bad key, spend limit, refusals) fails the same way every
  time and is raised at once.

These are retries of one call to the provider. Retries after a validation
error are a different loop, in ExtractionService, with a different prompt.
"""

import logging
import random
import time
from collections.abc import Callable
from typing import Any

from specsmith.llm.base import LLMProvider, LLMResponse, LLMUnavailableError

logger = logging.getLogger(__name__)

# Each delay is scaled by a factor in [1 - JITTER, 1].
JITTER = 0.25


class RetryingProvider:
    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_retries: int,
        base_delay_seconds: float,
        max_delay_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if not 0 < base_delay_seconds <= max_delay_seconds:
            raise ValueError(
                "delays must satisfy 0 < base_delay_seconds <= max_delay_seconds"
            )
        self._inner = inner
        self.model = inner.model
        self._max_retries = max_retries
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds
        self._sleep = sleep
        self._rand = rand

    def extract(self, prompt: str, schema: dict[str, Any]) -> LLMResponse:
        retries = 0
        while True:
            try:
                return self._inner.extract(prompt, schema)
            except LLMUnavailableError as exc:
                if not exc.retryable or retries >= self._max_retries:
                    raise
                delay = self._delay(retries, exc.retry_after_seconds)
                if delay is None:
                    raise
                retries += 1
                logger.warning(
                    "llm call failed, retrying",
                    extra={
                        "event": "llm_retry",
                        "retry": retries,
                        "max_retries": self._max_retries,
                        "delay_seconds": round(delay, 3),
                        "error": str(exc),
                    },
                )
                self._sleep(delay)

    def _delay(self, retries_so_far: int, retry_after: float | None) -> float | None:
        if retry_after is not None:
            return retry_after if retry_after <= self._max_delay else None
        backoff = min(self._base_delay * 2**retries_so_far, self._max_delay)
        return backoff * (1 - JITTER * self._rand())
