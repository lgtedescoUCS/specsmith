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
    # The configured model. Recorded for runs that fail before any response
    # arrives (a timeout on the first attempt) and so have no response.model.
    model: str

    def extract(self, prompt: str, schema: dict[str, Any]) -> LLMResponse:
        """Ask the model for output conforming to the JSON `schema`.

        Raises LLMTimeoutError when the model does not answer in time,
        LLMUnavailableError when the provider cannot be reached or returns
        an error status, and LLMError when the model answered but the answer
        is unusable (refused, cut off, empty).
        """
        ...


class LLMError(Exception):
    """The provider failed in a way that retrying the same call won't fix."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""


class LLMUnavailableError(LLMError):
    """The provider could not be reached or rejected the call (auth, rate
    limit, server error). A failure of the upstream service, not of the
    model's answer to this requirement."""
