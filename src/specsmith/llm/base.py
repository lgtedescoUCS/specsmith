"""Vendor-neutral contract between the service layer and any LLM.

Nothing here imports a vendor SDK. The service layer depends on this module
only; concrete providers (fake, Anthropic) implement LLMProvider.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMResponse:
    """What the model returned for one call.

    `content` is the raw text, not parsed JSON: invalid JSON and invalid
    fields then go through the same validation path and the same retry.
    Latency is not here on purpose: the caller measures it, because a call
    that times out has no response but still needs a latency on record.
    """

    content: str
    # The model that actually served the request, as reported by the
    # provider. It can differ from the one configured.
    model: str
    input_tokens: int
    output_tokens: int


class LLMProvider(Protocol):
    def extract(self, prompt: str, schema: dict[str, Any]) -> LLMResponse:
        """Ask the model for output conforming to the JSON `schema`.

        Raises LLMTimeoutError when the model does not answer in time and
        LLMError for any other failure that a retry would not fix.
        """
        ...


class LLMError(Exception):
    """The provider failed in a way that retrying the same call won't fix."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""
