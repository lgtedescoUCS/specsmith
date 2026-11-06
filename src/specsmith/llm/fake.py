"""In-memory LLMProvider for tests. No network, no tokens spent.

Each call consumes the next scripted item: a string becomes the response
content, an LLMResponse is returned as is, and an exception is raised.
Every call is recorded, so tests can assert on the prompts that were sent.
"""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from specsmith.llm.base import LLMResponse

FAKE_MODEL = "fake-model"

ScriptedItem = str | LLMResponse | Exception


@dataclass(frozen=True)
class FakeCall:
    prompt: str
    schema: dict[str, Any]


class FakeProviderExhaustedError(AssertionError):
    """The code under test made more calls than the test scripted.

    An AssertionError, not an LLMError, so the service cannot mistake it for
    a provider failure and swallow it: it is a bug in the test or the code.
    """


class FakeProvider:
    def __init__(
        self,
        responses: Iterable[ScriptedItem],
        *,
        model: str = FAKE_MODEL,
        input_tokens: int = 100,
        output_tokens: int = 50,
    ) -> None:
        self.model = model
        self._responses: deque[ScriptedItem] = deque(responses)
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls: list[FakeCall] = []

    def queue(self, *items: ScriptedItem) -> None:
        """Script more answers, for fakes built before the test knows them
        (such as the one the test suite injects into the app)."""
        self._responses.extend(items)

    def extract(self, prompt: str, schema: dict[str, Any]) -> LLMResponse:
        self.calls.append(FakeCall(prompt=prompt, schema=schema))
        if not self._responses:
            raise FakeProviderExhaustedError(
                f"FakeProvider got call #{len(self.calls)} "
                "but has no scripted responses left"
            )
        item = self._responses.popleft()
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(
            content=item,
            model=self.model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )
