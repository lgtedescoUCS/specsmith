"""Validation rules of the domain models. No database, no network."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from specsmith.domain.models import (
    Confidence,
    Constraint,
    ConstraintKind,
    Spec,
    StoredSpec,
)


def valid_payload() -> dict[str, Any]:
    return {
        "objective": "Increase tempering line throughput",
        "current_state": {"throughput": "400 plates/day"},
        "target_state": {"throughput": "650 plates/day"},
        "constraints": [
            {"kind": "space", "value": "12 m of available floor space"},
            {"kind": "deadline", "value": "operational by March"},
        ],
        "open_questions": ["How many shifts per day?"],
        "confidence": "medium",
    }


def assert_rejected(payload: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Spec.model_validate(payload)
    locations = [error["loc"] for error in exc_info.value.errors()]
    assert any(loc[0] == field for loc in locations), locations


def test_valid_payload_parses_into_typed_spec() -> None:
    spec = Spec.model_validate(valid_payload())

    assert spec.confidence is Confidence.MEDIUM
    assert spec.constraints[0] == Constraint(
        kind=ConstraintKind.SPACE, value="12 m of available floor space"
    )
    assert spec.open_questions == ["How many shifts per day?"]


def test_only_objective_and_confidence_are_required() -> None:
    spec = Spec.model_validate({"objective": "Reduce scrap", "confidence": "low"})

    assert spec.current_state == {}
    assert spec.target_state == {}
    assert spec.constraints == []
    assert spec.open_questions == []


@pytest.mark.parametrize("field", ["objective", "confidence"])
def test_required_field_missing_is_rejected(field: str) -> None:
    payload = valid_payload()
    del payload[field]

    assert_rejected(payload, field)


def test_unknown_field_is_rejected() -> None:
    payload = valid_payload() | {"priority": "high"}

    assert_rejected(payload, "priority")


def test_unknown_field_inside_constraint_is_rejected() -> None:
    payload = valid_payload()
    payload["constraints"][0]["unit"] = "m"

    assert_rejected(payload, "constraints")


@pytest.mark.parametrize("objective", ["", "   "])
def test_blank_objective_is_rejected(objective: str) -> None:
    payload = valid_payload() | {"objective": objective}

    assert_rejected(payload, "objective")


def test_objective_longer_than_300_chars_is_rejected() -> None:
    payload = valid_payload() | {"objective": "x" * 301}

    assert_rejected(payload, "objective")


def test_surrounding_whitespace_is_stripped() -> None:
    payload = valid_payload() | {
        "objective": "  Reduce scrap  ",
        "open_questions": ["  Which alloy?  "],
    }

    spec = Spec.model_validate(payload)

    assert spec.objective == "Reduce scrap"
    assert spec.open_questions == ["Which alloy?"]


def test_unknown_constraint_kind_is_rejected() -> None:
    payload = valid_payload()
    payload["constraints"][0]["kind"] = "colour"

    assert_rejected(payload, "constraints")


@pytest.mark.parametrize("value", ["", "x" * 301])
def test_constraint_value_out_of_bounds_is_rejected(value: str) -> None:
    payload = valid_payload()
    payload["constraints"][0]["value"] = value

    assert_rejected(payload, "constraints")


def test_unknown_confidence_is_rejected() -> None:
    payload = valid_payload() | {"confidence": "certain"}

    assert_rejected(payload, "confidence")


def test_ten_open_questions_are_accepted() -> None:
    payload = valid_payload() | {"open_questions": [f"Q{i}?" for i in range(10)]}

    assert len(Spec.model_validate(payload).open_questions) == 10


def test_more_than_ten_open_questions_are_rejected() -> None:
    payload = valid_payload() | {"open_questions": [f"Q{i}?" for i in range(11)]}

    assert_rejected(payload, "open_questions")


def test_blank_open_question_is_rejected() -> None:
    payload = valid_payload() | {"open_questions": ["Which alloy?", "  "]}

    assert_rejected(payload, "open_questions")


def test_stored_spec_adds_identity_and_serialises_to_json() -> None:
    spec_id, requirement_id = uuid4(), uuid4()
    stored = StoredSpec.model_validate(
        valid_payload()
        | {
            "id": spec_id,
            "requirement_id": requirement_id,
            "created_at": datetime(2026, 11, 6, tzinfo=UTC),
        }
    )

    dumped = stored.model_dump(mode="json")

    assert dumped["id"] == str(spec_id)
    assert dumped["requirement_id"] == str(requirement_id)
    assert dumped["constraints"][0] == {
        "kind": "space",
        "value": "12 m of available floor space",
    }
    assert dumped["confidence"] == "medium"
