"""Schema and loader of the evaluation dataset. No database, no network.

The answer keys below are synthetic and exist only to exercise the schema;
they are not annotations of eval/dataset.jsonl.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from specsmith.domain.models import Confidence, ConstraintKind
from specsmith.evaluation.dataset import (
    DatasetError,
    EvalCase,
    Expected,
    load_dataset,
)


def expected_payload() -> dict[str, Any]:
    return {
        "objective_keywords": ["widget", ["speed", "rate"]],
        "constraint_kinds": ["budget", "space"],
        "min_open_questions": 2,
        "confidence": "medium",
        "must_ask_topics": {"colour": ["colour", "color"]},
        "must_not_ask_topics": {"budget": ["what is the budget"]},
        "forbidden_terms": ["gizmo"],
    }


def case_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "id": "001",
        "lang": "en",
        "tags": ["synthetic"],
        "raw_text": "Make widgets faster.",
        "expected": expected_payload(),
    }
    payload.update(overrides)
    return payload


def write_lines(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def test_annotated_case_parses() -> None:
    case = EvalCase.model_validate(case_payload())

    assert case.annotated
    assert case.expected is not None
    assert case.expected.objective_keywords == ["widget", ["speed", "rate"]]
    assert case.expected.constraint_kinds == [
        ConstraintKind.BUDGET,
        ConstraintKind.SPACE,
    ]
    assert case.expected.confidence is Confidence.MEDIUM


def test_empty_expected_means_unannotated() -> None:
    case = EvalCase.model_validate(case_payload(expected={}))

    assert not case.annotated
    assert case.expected is None


def test_null_expected_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"use \{\}"):
        EvalCase.model_validate(case_payload(expected=None))


def test_misspelled_field_is_rejected() -> None:
    payload = expected_payload()
    payload["must_ask_topic"] = payload.pop("must_ask_topics")

    with pytest.raises(ValidationError) as exc_info:
        Expected.model_validate(payload)
    locations = {error["loc"][0] for error in exc_info.value.errors()}
    assert locations == {"must_ask_topic", "must_ask_topics"}


@pytest.mark.parametrize("field", list(expected_payload()))
def test_every_expected_field_is_required(field: str) -> None:
    # A forgotten field must not score as "nothing expected".
    payload = expected_payload()
    del payload[field]

    with pytest.raises(ValidationError):
        Expected.model_validate(payload)


def test_empty_lists_and_maps_are_allowed_when_explicit() -> None:
    payload = expected_payload() | {
        "objective_keywords": [],
        "constraint_kinds": [],
        "min_open_questions": 0,
        "must_ask_topics": {},
        "must_not_ask_topics": {},
        "forbidden_terms": [],
    }

    expected = Expected.model_validate(payload)

    assert expected.must_ask_topics == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("constraint_kinds", ["budget", "budget"]),
        ("constraint_kinds", ["cost"]),
        ("min_open_questions", -1),
        ("min_open_questions", 11),
        ("confidence", "certain"),
        ("objective_keywords", ["  "]),
        ("objective_keywords", [[]]),
        ("must_ask_topics", {"colour": []}),
        ("must_ask_topics", {"colour": [" "]}),
        ("forbidden_terms", [""]),
    ],
)
def test_invalid_expected_value_is_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        Expected.model_validate(expected_payload() | {field: value})


def test_topic_in_both_lists_is_rejected() -> None:
    payload = expected_payload() | {
        "must_ask_topics": {"budget": ["budget"]},
        "must_not_ask_topics": {"budget": ["budget"]},
    }

    with pytest.raises(ValidationError, match="budget"):
        Expected.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("id", "1"), ("id", "0001"), ("lang", "es"), ("raw_text", "   ")],
)
def test_invalid_case_value_is_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate(case_payload(**{field: value}))


def test_load_dataset_reads_cases_in_order(tmp_path: Path) -> None:
    path = write_lines(
        tmp_path,
        json.dumps(case_payload(id="001")),
        "",
        json.dumps(case_payload(id="002", lang="pt", raw_text="Têmpera.")),
    )

    cases = load_dataset(path)

    assert [case.id for case in cases] == ["001", "002"]
    assert cases[1].raw_text == "Têmpera."


def test_load_dataset_names_the_line_of_invalid_json(tmp_path: Path) -> None:
    path = write_lines(tmp_path, json.dumps(case_payload()), "{not json")

    with pytest.raises(DatasetError, match=r"dataset\.jsonl:2: invalid JSON"):
        load_dataset(path)


def test_load_dataset_names_the_line_of_an_invalid_case(tmp_path: Path) -> None:
    path = write_lines(
        tmp_path, json.dumps(case_payload()), json.dumps(case_payload(id="x"))
    )

    with pytest.raises(DatasetError, match=r"dataset\.jsonl:2:"):
        load_dataset(path)


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    line = json.dumps(case_payload(id="007"))
    path = write_lines(tmp_path, line, line)

    with pytest.raises(DatasetError, match="duplicate id '007'.*line 1"):
        load_dataset(path)


def test_load_dataset_rejects_an_empty_file(tmp_path: Path) -> None:
    path = write_lines(tmp_path, "")

    with pytest.raises(DatasetError, match="no cases"):
        load_dataset(path)
