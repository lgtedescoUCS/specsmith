"""Extraction loop: prompt -> LLM -> validate -> retry with the error.

This module depends on the LLMProvider protocol only, never on a vendor SDK,
and on no database: persisting the outcome is the caller's job.
"""

import time
from dataclasses import dataclass

from pydantic import ValidationError

from specsmith.domain.errors import ExtractionFailedError, ExtractionTimeoutError
from specsmith.domain.models import ExtractionMeta, Spec
from specsmith.llm.base import LLMError, LLMProvider, LLMTimeoutError
from specsmith.llm.prompts import Prompt
from specsmith.services.spec_draft import SPEC_DRAFT_SCHEMA, parse_spec


@dataclass(frozen=True)
class ExtractionResult:
    spec: Spec
    meta: ExtractionMeta


class ExtractionService:
    def __init__(
        self, provider: LLMProvider, prompt: Prompt, *, max_retries: int
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        self._provider = provider
        self._prompt = prompt
        self._max_retries = max_retries

    def extract(self, raw_text: str) -> ExtractionResult:
        """Raises ExtractionFailedError or ExtractionTimeoutError, each
        carrying the metadata of the attempts made."""
        prompt = self._prompt.render(raw_text)
        model = self._provider.model
        input_tokens = output_tokens = attempts = 0
        started = time.perf_counter()

        def meta() -> ExtractionMeta:
            return ExtractionMeta(
                model=model,
                prompt_version=self._prompt.version,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round((time.perf_counter() - started) * 1000),
                attempts=attempts,
            )

        while True:
            attempts += 1
            try:
                response = self._provider.extract(prompt, SPEC_DRAFT_SCHEMA)
            # Provider failures are not retried here: a timeout or a refusal
            # would most likely repeat. Backoff for transient errors is a
            # provider concern (phase 4).
            except LLMTimeoutError as exc:
                raise ExtractionTimeoutError(f"LLM timed out: {exc}", meta()) from exc
            except LLMError as exc:
                raise ExtractionFailedError(f"LLM failed: {exc}", meta()) from exc

            model = response.model
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            try:
                spec = parse_spec(response.content)
            except ValidationError as exc:
                errors = describe_validation_error(exc)
                if attempts > self._max_retries:
                    raise ExtractionFailedError(
                        f"Output failed validation after {attempts} attempt(s):\n"
                        f"{errors}",
                        meta(),
                    ) from exc
                # Each retry is a fresh single-message request: the main
                # prompt plus the rejected answer and its errors, worded by
                # the same prompt version. Not a growing conversation.
                prompt = self._prompt.render_retry(
                    raw_text, previous_answer=response.content, errors=errors
                )
                continue
            return ExtractionResult(spec=spec, meta=meta())


def describe_validation_error(error: ValidationError) -> str:
    """One line per error, `- location: message`, without the input values
    (the model already has its own answer just above)."""
    lines = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)
