"""The fake LLM provider used by the rest of the suite. Offline."""

import pytest

from specsmith.llm.base import LLMError, LLMProvider, LLMResponse, LLMTimeoutError
from specsmith.llm.fake import (
    FAKE_MODEL,
    FakeCall,
    FakeProvider,
    FakeProviderExhaustedError,
)

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def test_string_item_becomes_response_with_default_metadata() -> None:
    provider = FakeProvider(['{"objective": "x"}'])

    response = provider.extract("prompt", SCHEMA)

    assert response == LLMResponse(
        content='{"objective": "x"}',
        model=FAKE_MODEL,
        input_tokens=100,
        output_tokens=50,
    )


def test_token_counts_are_configurable() -> None:
    provider = FakeProvider(["{}"], input_tokens=7, output_tokens=3)

    response = provider.extract("prompt", SCHEMA)

    assert (response.input_tokens, response.output_tokens) == (7, 3)


def test_llm_response_item_is_returned_unchanged() -> None:
    scripted = LLMResponse(
        content="{}", model="other-model", input_tokens=1, output_tokens=2
    )
    provider = FakeProvider([scripted])

    assert provider.extract("prompt", SCHEMA) is scripted


def test_responses_are_consumed_in_order() -> None:
    provider = FakeProvider(["first", "second"])

    contents = [provider.extract("p", SCHEMA).content for _ in range(2)]

    assert contents == ["first", "second"]


@pytest.mark.parametrize(
    "error", [LLMTimeoutError("too slow"), LLMError("refused")]
)
def test_exception_item_is_raised(error: LLMError) -> None:
    provider = FakeProvider([error, "{}"])

    with pytest.raises(type(error)) as exc_info:
        provider.extract("prompt", SCHEMA)

    assert exc_info.value is error
    # The failed call still consumed its item; the next one is served.
    assert provider.extract("prompt", SCHEMA).content == "{}"


def test_every_call_is_recorded_including_failures() -> None:
    provider = FakeProvider([LLMTimeoutError("too slow"), "{}"])

    with pytest.raises(LLMTimeoutError):
        provider.extract("first prompt", SCHEMA)
    provider.extract("second prompt", SCHEMA)

    assert provider.calls == [
        FakeCall(prompt="first prompt", schema=SCHEMA),
        FakeCall(prompt="second prompt", schema=SCHEMA),
    ]


def test_running_out_of_responses_fails_loudly() -> None:
    provider = FakeProvider(["{}"])
    provider.extract("prompt", SCHEMA)

    with pytest.raises(FakeProviderExhaustedError, match="call #2"):
        provider.extract("prompt", SCHEMA)


def test_exhaustion_is_not_mistaken_for_a_provider_failure() -> None:
    # Code that catches LLMError must not swallow a scripting bug.
    assert not issubclass(FakeProviderExhaustedError, LLMError)


def test_timeout_is_a_kind_of_llm_error() -> None:
    assert issubclass(LLMTimeoutError, LLMError)


def test_fake_satisfies_the_provider_protocol() -> None:
    # Checked statically by a type checker; at runtime this documents intent.
    provider: LLMProvider = FakeProvider(["{}"])

    assert provider.extract("prompt", SCHEMA).content == "{}"
