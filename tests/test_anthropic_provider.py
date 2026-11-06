"""AnthropicProvider through the real SDK, over a mock HTTP transport.

No network and no tokens: every request is answered by a local handler.
Going through the SDK (instead of stubbing messages.create) checks the
request body that would go on the wire and the SDK's own error mapping.
"""

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest

from specsmith.llm.anthropic import DEFAULT_MAX_TOKENS, AnthropicProvider
from specsmith.llm.base import (
    LLMError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
)
from specsmith.services.spec_draft import SPEC_DRAFT_SCHEMA

MODEL = "claude-sonnet-5-5"

Handler = Callable[[httpx2.Request], httpx2.Response]


def message_body(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": '{"objective": "x"}'}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 412, "output_tokens": 287},
    } | overrides


def provider_with(handler: Handler) -> tuple[AnthropicProvider, list[httpx2.Request]]:
    requests: list[httpx2.Request] = []

    def recording(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return handler(request)

    client = anthropic.Anthropic(
        api_key="test-key-not-real",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(
            transport=httpx2.MockTransport(recording)
        ),
    )
    return AnthropicProvider(client, model=MODEL), requests


def respond(status: int = 200, **overrides: Any) -> Handler:
    body = message_body(**overrides) if status == 200 else overrides
    return lambda request: httpx2.Response(status, json=body)


def api_error(status: int, error_type: str, message: str) -> Handler:
    return respond(status, type="error", error={"type": error_type, "message": message})


# --- request ---------------------------------------------------------------


def test_request_uses_structured_output_with_the_transformed_schema() -> None:
    provider, requests = provider_with(respond())

    provider.extract("the prompt", SPEC_DRAFT_SCHEMA)

    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/messages"
    assert body["model"] == MODEL
    assert body["max_tokens"] == DEFAULT_MAX_TOKENS
    assert body["messages"] == [{"role": "user", "content": "the prompt"}]
    assert body["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": anthropic.transform_schema(SPEC_DRAFT_SCHEMA),
        }
    }
    # No forced tool use: it is rejected on this model family.
    assert "tools" not in body and "tool_choice" not in body


def test_unsupported_schema_keywords_are_not_sent() -> None:
    provider, requests = provider_with(respond())

    provider.extract("p", SPEC_DRAFT_SCHEMA)

    sent = json.dumps(json.loads(requests[0].content)["output_config"])
    for keyword in ('"minLength"', '"maxLength"', '"maxItems"'):
        assert keyword not in sent
    # They survive as descriptions the model can read.
    assert "maxItems: 10" in sent


def test_the_service_schema_itself_is_not_mutated() -> None:
    before = json.dumps(SPEC_DRAFT_SCHEMA, sort_keys=True)
    provider, _ = provider_with(respond())

    provider.extract("p", SPEC_DRAFT_SCHEMA)

    assert json.dumps(SPEC_DRAFT_SCHEMA, sort_keys=True) == before


# --- response --------------------------------------------------------------


def test_text_and_usage_become_an_llm_response() -> None:
    provider, _ = provider_with(respond())

    assert provider.extract("p", SPEC_DRAFT_SCHEMA) == LLMResponse(
        content='{"objective": "x"}',
        model=MODEL,
        input_tokens=412,
        output_tokens=287,
    )


def test_model_is_the_one_that_served_the_request() -> None:
    provider, _ = provider_with(respond(model="claude-opus-5-5"))

    assert provider.extract("p", SPEC_DRAFT_SCHEMA).model == "claude-opus-5-5"
    assert provider.model == MODEL


def test_thinking_blocks_are_skipped() -> None:
    provider, _ = provider_with(
        respond(
            content=[
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "{}"},
            ]
        )
    )

    assert provider.extract("p", SPEC_DRAFT_SCHEMA).content == "{}"


def test_invalid_json_text_is_passed_through_for_the_service_to_reject() -> None:
    # Validation and retry belong to the service, not the provider.
    provider, _ = provider_with(respond(content=[{"type": "text", "text": "oops"}]))

    assert provider.extract("p", SPEC_DRAFT_SCHEMA).content == "oops"


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "stop_reason": "refusal",
                "stop_details": {
                    "type": "refusal",
                    "category": "cyber",
                    "explanation": "x",
                },
            },
            r"declined the request \(category: cyber\)",
        ),
        ({"stop_reason": "max_tokens"}, "cut off at max_tokens=16000"),
        ({"content": []}, "no text content"),
    ],
    ids=["refusal", "max-tokens", "no-text"],
)
def test_unusable_responses_are_llm_errors(
    overrides: dict[str, Any], message: str
) -> None:
    provider, _ = provider_with(respond(**overrides))

    with pytest.raises(LLMError, match=message) as exc_info:
        provider.extract("p", SPEC_DRAFT_SCHEMA)

    # The model answered: this is about its answer, not the upstream service.
    assert not isinstance(exc_info.value, (LLMTimeoutError, LLMUnavailableError))


# --- errors ----------------------------------------------------------------


def test_timeout_becomes_llm_timeout_error() -> None:
    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("read timed out", request=request)

    provider, requests = provider_with(slow)

    with pytest.raises(LLMTimeoutError, match="timed out"):
        provider.extract("p", SPEC_DRAFT_SCHEMA)

    assert len(requests) == 1


def test_connection_failure_is_unavailable_not_a_timeout() -> None:
    def unreachable(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    provider, _ = provider_with(unreachable)

    with pytest.raises(LLMUnavailableError, match="Could not reach") as exc_info:
        provider.extract("p", SPEC_DRAFT_SCHEMA)

    assert not isinstance(exc_info.value, LLMTimeoutError)


@pytest.mark.parametrize(
    "status, error_type",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (529, "overloaded_error"),
    ],
)
def test_api_errors_mean_the_provider_is_unavailable(
    status: int, error_type: str
) -> None:
    provider, requests = provider_with(api_error(status, error_type, "nope"))

    with pytest.raises(LLMUnavailableError, match=f"returned {status}"):
        provider.extract("p", SPEC_DRAFT_SCHEMA)

    assert len(requests) == 1


def test_error_messages_never_contain_the_api_key() -> None:
    provider, _ = provider_with(api_error(401, "authentication_error", "bad key"))

    with pytest.raises(LLMError) as exc_info:
        provider.extract("p", SPEC_DRAFT_SCHEMA)

    assert "test-key-not-real" not in str(exc_info.value)


# --- construction ----------------------------------------------------------


def test_create_disables_sdk_retries_and_sets_the_timeout() -> None:
    provider = AnthropicProvider.create(
        api_key="test-key-not-real", model=MODEL, timeout_seconds=30
    )

    assert provider.model == MODEL
    assert provider._client.max_retries == 0
    assert provider._client.timeout == 30


@pytest.mark.parametrize("api_key", ["", None])
def test_create_requires_an_api_key(api_key: Any) -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider.create(api_key=api_key, model=MODEL, timeout_seconds=30)
