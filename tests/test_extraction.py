"""Extraction loop with the fake provider. Offline, no tokens spent."""

import json
from collections.abc import Iterator
from typing import Any

import pytest

from specsmith.domain.errors import ExtractionFailedError, ExtractionTimeoutError
from specsmith.domain.models import ExtractionMeta, Spec
from specsmith.llm.base import LLMError, LLMResponse, LLMTimeoutError
from specsmith.llm.fake import FAKE_MODEL, FakeProvider, ScriptedItem
from specsmith.llm.prompts import load_prompt
from specsmith.services import extraction
from specsmith.services.extraction import ExtractionService
from specsmith.services.spec_draft import SPEC_DRAFT_SCHEMA

RAW_TEXT = "Client wants 650 plates/day on the tempering line, budget {1.2M}."
PROMPT = load_prompt("v1")


def draft(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "objective": "Increase tempering line throughput",
        "current_state": [{"name": "throughput", "value": "400 plates/day"}],
        "target_state": [{"name": "throughput", "value": "650 plates/day"}],
        "constraints": [{"kind": "budget", "value": "approx. USD 1,200,000"}],
        "open_questions": ["How many shifts per day?"],
        "confidence": "medium",
    }
    return base | overrides


def as_json(**overrides: Any) -> str:
    return json.dumps(draft(**overrides))


def service(
    *responses: ScriptedItem, max_retries: int = 2
) -> tuple[ExtractionService, FakeProvider]:
    provider = FakeProvider(responses)
    return ExtractionService(provider, PROMPT, max_retries=max_retries), provider


# --- success ---------------------------------------------------------------


def test_valid_first_answer_becomes_a_spec() -> None:
    svc, provider = service(as_json())

    result = svc.extract(RAW_TEXT)

    assert result.spec == Spec.model_validate(
        {
            "objective": "Increase tempering line throughput",
            "current_state": {"throughput": "400 plates/day"},
            "target_state": {"throughput": "650 plates/day"},
            "constraints": [{"kind": "budget", "value": "approx. USD 1,200,000"}],
            "open_questions": ["How many shifts per day?"],
            "confidence": "medium",
        }
    )
    assert result.meta.attempts == 1
    assert len(provider.calls) == 1


def test_first_call_sends_rendered_prompt_and_draft_schema() -> None:
    svc, provider = service(as_json())

    svc.extract(RAW_TEXT)

    assert provider.calls[0].prompt == PROMPT.render(RAW_TEXT)
    assert provider.calls[0].schema == SPEC_DRAFT_SCHEMA


def test_meta_records_model_prompt_version_and_tokens() -> None:
    svc, _ = service(as_json())

    meta = svc.extract(RAW_TEXT).meta

    assert (meta.model, meta.prompt_version) == (FAKE_MODEL, "v1")
    assert (meta.input_tokens, meta.output_tokens) == (100, 50)


def test_meta_records_the_model_that_actually_answered() -> None:
    served = LLMResponse(
        content=as_json(), model="served-model", input_tokens=1, output_tokens=1
    )
    svc, _ = service(served)

    assert svc.extract(RAW_TEXT).meta.model == "served-model"


def test_latency_covers_the_whole_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    clock: Iterator[float] = iter([10.0, 11.25])
    monkeypatch.setattr(extraction.time, "perf_counter", lambda: next(clock))
    svc, _ = service(as_json())

    assert svc.extract(RAW_TEXT).meta.latency_ms == 1250


def test_empty_lists_are_valid() -> None:
    svc, _ = service(
        as_json(current_state=[], target_state=[], constraints=[], open_questions=[])
    )

    spec = svc.extract(RAW_TEXT).spec

    assert (spec.current_state, spec.constraints, spec.open_questions) == ({}, [], [])


# --- retry on invalid output -----------------------------------------------


@pytest.mark.parametrize(
    "invalid, expected_error",
    [
        pytest.param("not json at all", "- (root): Invalid JSON", id="invalid-json"),
        pytest.param(
            json.dumps({k: v for k, v in draft().items() if k != "confidence"}),
            "- confidence: Field required",
            id="missing-field",
        ),
        pytest.param(
            as_json(invented="x"), "- invented: Extra inputs", id="extra-field"
        ),
        pytest.param(
            as_json(confidence="certain"), "- confidence: Input should be",
            id="bad-enum",
        ),
        pytest.param(
            as_json(constraints=[{"kind": "colour", "value": "red"}]),
            "- constraints.0.kind: Input should be",
            id="bad-constraint-kind",
        ),
        pytest.param(
            as_json(objective="x" * 301), "- objective: String should have at most",
            id="objective-too-long",
        ),
        pytest.param(
            as_json(objective="   "), "- objective: String should have at least",
            id="blank-objective",
        ),
        pytest.param(
            as_json(open_questions=[f"Q{i}?" for i in range(11)]),
            "- open_questions: List should have at most 10",
            id="too-many-questions",
        ),
        pytest.param(
            as_json(current_state={"throughput": "400"}),
            "- current_state: Input should be a valid array",
            id="state-as-dict",
        ),
    ],
)
def test_invalid_answer_is_retried_with_its_errors(
    invalid: str, expected_error: str
) -> None:
    svc, provider = service(invalid, as_json())

    result = svc.extract(RAW_TEXT)

    assert result.meta.attempts == 2
    retry_prompt = provider.calls[1].prompt
    assert retry_prompt.startswith(PROMPT.render(RAW_TEXT))
    assert f"<previous_answer>\n{invalid}\n</previous_answer>" in retry_prompt
    assert expected_error in retry_prompt


def test_retry_prompt_is_the_versioned_retry_template() -> None:
    invalid = as_json(confidence="certain")
    svc, provider = service(invalid, as_json())

    svc.extract(RAW_TEXT)

    errors = provider.calls[1].prompt.split("validation errors:\n")[1].split("\n\n")[0]
    assert provider.calls[1].prompt == PROMPT.render_retry(
        RAW_TEXT, previous_answer=invalid, errors=errors
    )


def test_each_retry_carries_only_the_latest_answer() -> None:
    first, second = "first bad answer", "second bad answer"
    svc, provider = service(first, second, as_json())

    svc.extract(RAW_TEXT)

    assert first not in provider.calls[2].prompt
    assert second in provider.calls[2].prompt


def test_tokens_are_summed_over_attempts() -> None:
    svc, _ = service(
        LLMResponse("bad", FAKE_MODEL, input_tokens=10, output_tokens=1),
        LLMResponse(as_json(), FAKE_MODEL, input_tokens=20, output_tokens=2),
    )

    meta = svc.extract(RAW_TEXT).meta

    assert (meta.input_tokens, meta.output_tokens, meta.attempts) == (30, 3, 2)


# --- duplicate state names (decision B) ------------------------------------


@pytest.mark.parametrize("field", ["current_state", "target_state"])
def test_duplicate_state_name_is_retried(field: str) -> None:
    duplicated = as_json(
        **{
            field: [
                {"name": "throughput", "value": "400 plates/day"},
                {"name": "throughput", "value": "450 plates/day"},
            ]
        }
    )
    svc, provider = service(duplicated, as_json())

    result = svc.extract(RAW_TEXT)

    assert result.meta.attempts == 2
    assert (
        f"- {field}: Value error, duplicate name 'throughput'; "
        "each name must appear once"
    ) in provider.calls[1].prompt


def test_names_differing_only_by_whitespace_are_duplicates() -> None:
    duplicated = as_json(
        current_state=[
            {"name": "throughput", "value": "400"},
            {"name": "  throughput ", "value": "450"},
        ]
    )
    svc, provider = service(duplicated, as_json())

    svc.extract(RAW_TEXT)

    assert "duplicate name 'throughput'" in provider.calls[1].prompt


def test_every_duplicated_name_is_reported_once() -> None:
    duplicated = as_json(
        current_state=[
            {"name": "a", "value": "1"},
            {"name": "b", "value": "2"},
            {"name": "a", "value": "3"},
            {"name": "b", "value": "4"},
            {"name": "a", "value": "5"},
        ]
    )
    svc, provider = service(duplicated, as_json())

    svc.extract(RAW_TEXT)

    assert "duplicate name 'a', 'b'; each name" in provider.calls[1].prompt


def test_names_differing_in_case_are_distinct() -> None:
    # They are different dict keys, so nothing would be lost.
    svc, _ = service(
        as_json(
            current_state=[
                {"name": "Throughput", "value": "400"},
                {"name": "throughput", "value": "450"},
            ]
        )
    )

    result = svc.extract(RAW_TEXT)

    assert result.spec.current_state == {"Throughput": "400", "throughput": "450"}
    assert result.meta.attempts == 1


def test_persistent_duplicates_exhaust_the_retries() -> None:
    duplicated = as_json(
        target_state=[{"name": "x", "value": "1"}, {"name": "x", "value": "2"}]
    )
    svc, provider = service(duplicated, duplicated, duplicated)

    with pytest.raises(ExtractionFailedError, match="duplicate name 'x'"):
        svc.extract(RAW_TEXT)

    assert len(provider.calls) == 3


# --- giving up -------------------------------------------------------------


def test_gives_up_after_max_retries() -> None:
    svc, provider = service("bad", "bad", "bad", max_retries=2)

    with pytest.raises(ExtractionFailedError) as exc_info:
        svc.extract(RAW_TEXT)

    assert len(provider.calls) == 3
    error = exc_info.value
    assert error.reason.startswith("Output failed validation after 3 attempt(s):")
    assert "- (root): Invalid JSON" in error.reason
    assert error.meta == ExtractionMeta(
        model=FAKE_MODEL,
        prompt_version="v1",
        input_tokens=300,
        output_tokens=150,
        latency_ms=error.meta.latency_ms,
        attempts=3,
    )


def test_zero_retries_means_a_single_attempt() -> None:
    svc, provider = service("bad", max_retries=0)

    with pytest.raises(ExtractionFailedError) as exc_info:
        svc.extract(RAW_TEXT)

    assert len(provider.calls) == 1
    assert exc_info.value.meta.attempts == 1


def test_succeeding_on_the_last_allowed_attempt_is_a_success() -> None:
    svc, _ = service("bad", "bad", as_json(), max_retries=2)

    assert svc.extract(RAW_TEXT).meta.attempts == 3


def test_negative_max_retries_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        ExtractionService(FakeProvider([]), PROMPT, max_retries=-1)


# --- provider failures are not retried -------------------------------------


def test_timeout_is_not_retried_and_carries_meta() -> None:
    svc, provider = service(LLMTimeoutError("no answer in 30 s"), as_json())

    with pytest.raises(ExtractionTimeoutError) as exc_info:
        svc.extract(RAW_TEXT)

    assert len(provider.calls) == 1
    meta = exc_info.value.meta
    # No response arrived: the configured model and zero tokens are recorded.
    assert (meta.model, meta.input_tokens, meta.output_tokens, meta.attempts) == (
        FAKE_MODEL,
        0,
        0,
        1,
    )
    assert "no answer in 30 s" in exc_info.value.reason


def test_timeout_on_a_retry_keeps_the_earlier_attempts() -> None:
    svc, _ = service(
        LLMResponse("bad", "served-model", input_tokens=10, output_tokens=5),
        LLMTimeoutError("slow"),
    )

    with pytest.raises(ExtractionTimeoutError) as exc_info:
        svc.extract(RAW_TEXT)

    meta = exc_info.value.meta
    assert (meta.model, meta.input_tokens, meta.output_tokens, meta.attempts) == (
        "served-model",
        10,
        5,
        2,
    )


def test_other_provider_failure_is_not_retried() -> None:
    svc, provider = service(LLMError("model declined"), as_json())

    with pytest.raises(ExtractionFailedError, match="model declined") as exc_info:
        svc.extract(RAW_TEXT)

    assert len(provider.calls) == 1
    assert exc_info.value.meta.attempts == 1


def test_timeout_is_not_reported_as_a_plain_failure() -> None:
    # The API maps them to different statuses (504 vs 422).
    assert not issubclass(ExtractionTimeoutError, ExtractionFailedError)


# --- the schema sent to the model (decision B) -----------------------------


def _objects(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _objects(value)


def test_every_object_in_the_schema_forbids_additional_properties() -> None:
    objects = list(_objects(SPEC_DRAFT_SCHEMA))

    assert len(objects) == 3  # SpecDraft, StateEntry, Constraint
    assert all(obj.get("additionalProperties") is False for obj in objects)


def test_every_schema_field_is_required() -> None:
    for obj in _objects(SPEC_DRAFT_SCHEMA):
        assert set(obj["required"]) == set(obj["properties"])
