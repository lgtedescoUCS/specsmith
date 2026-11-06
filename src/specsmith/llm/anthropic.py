"""LLMProvider backed by the Anthropic Messages API.

The only module that imports the anthropic SDK. Everything it returns or
raises is vendor-neutral (LLMResponse, LLMError, LLMTimeoutError).

Structured output uses `output_config.format` with a JSON schema, which
constrains the model's answer to that schema:
https://platform.claude.com/docs/en/build-with-claude/structured-outputs
"""

from typing import Any

import anthropic

from specsmith.llm.base import (
    LLMError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
)

# Non-streaming requests stay well under the SDK's HTTP timeout at this
# size, and it leaves room for thinking plus a spec-sized JSON answer.
DEFAULT_MAX_TOKENS = 16000


class AnthropicProvider:
    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = client
        self.model = model
        self._max_tokens = max_tokens

    @classmethod
    def create(
        cls, *, api_key: str, model: str, timeout_seconds: float
    ) -> "AnthropicProvider":
        if not api_key:
            raise ValueError("An Anthropic API key is required (ANTHROPIC_API_KEY)")
        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_seconds,
            # The SDK retries timeouts, 429 and 5xx by default, which would
            # stretch a 30 s timeout to 90 s before the client gets a 504.
            # Retrying with backoff is a deliberate phase 4 change.
            max_retries=0,
        )
        return cls(client, model=model)

    def extract(self, prompt: str, schema: dict[str, Any]) -> LLMResponse:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        # The API rejects keywords it cannot enforce, such
                        # as minLength and maxItems, and messages.create
                        # sends a raw schema unchanged. transform_schema
                        # moves them into descriptions; Pydantic in the
                        # service still enforces every one of them.
                        "schema": anthropic.transform_schema(schema),
                    }
                },
            )
        # APITimeoutError subclasses APIConnectionError, so it comes first.
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(f"Anthropic API timed out: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMUnavailableError(
                f"Anthropic API returned {exc.status_code}: {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailableError(
                f"Could not reach the Anthropic API: {exc}"
            ) from exc

        # Both stop reasons void the schema guarantee, and retrying the same
        # request would most likely end the same way, so neither is retried.
        if message.stop_reason == "refusal":
            category = message.stop_details.category if message.stop_details else None
            raise LLMError(f"Model declined the request (category: {category})")
        if message.stop_reason == "max_tokens":
            raise LLMError(
                f"Output cut off at max_tokens={self._max_tokens}; it may be "
                "incomplete JSON"
            )

        # Thinking blocks may precede the answer; the JSON is in text blocks.
        text = "".join(block.text for block in message.content if block.type == "text")
        if not text:
            raise LLMError(
                f"Response had no text content (stop_reason: {message.stop_reason})"
            )
        return LLMResponse(
            content=text,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
