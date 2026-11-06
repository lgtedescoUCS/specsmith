"""Schema and loader for eval/dataset.jsonl.

One JSON object per line:

    {"id": "001", "lang": "pt", "tags": [...], "raw_text": "...",
     "expected": {...}}

`expected` is the hand-written answer key. `{}` marks a case that has not
been annotated yet. Once annotated, every field must be present, even when
empty: a forgotten field would otherwise score as "nothing expected" and
pass silently. Unknown keys are rejected for the same reason, so a typo
such as `must_ask_topic` fails at load time.

Keywords, aliases and terms are matched against the (English) model output
case-insensitively, at the start of a word: "temper" matches "tempering"
but "heat" does not match "preheat". See scoring.py.
"""

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from specsmith.domain.models import MAX_OPEN_QUESTIONS, Confidence, ConstraintKind


class _EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


Term = Annotated[str, Field(min_length=1)]
# Any one alternative is enough for the keyword to count as present.
Alternatives = Annotated[list[Term], Field(min_length=1)]
Keyword = Term | Alternatives
# Topic name -> aliases; a question mentioning any alias covers the topic.
Topics = dict[Term, Alternatives]


class Expected(_EvalModel):
    objective_keywords: list[Keyword]
    constraint_kinds: list[ConstraintKind]
    # Bounded by what the output schema allows, so an unreachable minimum
    # is caught at load time rather than failing every run.
    min_open_questions: int = Field(ge=0, le=MAX_OPEN_QUESTIONS)
    confidence: Confidence
    must_ask_topics: Topics
    must_not_ask_topics: Topics
    forbidden_terms: list[Term]

    @field_validator("constraint_kinds")
    @classmethod
    def kinds_are_unique(cls, kinds: list[ConstraintKind]) -> list[ConstraintKind]:
        # Scored as a set; a repeated kind is an annotation slip.
        if len(set(kinds)) != len(kinds):
            raise ValueError("constraint kinds must not repeat")
        return kinds

    @model_validator(mode="after")
    def topics_do_not_overlap(self) -> "Expected":
        both = sorted(self.must_ask_topics.keys() & self.must_not_ask_topics.keys())
        if both:
            raise ValueError(
                f"topics in both must_ask_topics and must_not_ask_topics: {both}"
            )
        return self


class EvalCase(_EvalModel):
    id: str = Field(pattern=r"^\d{3}$")
    lang: Literal["pt", "en"]
    # Describe the input (what makes the case hard), never the answer.
    tags: list[Term]
    raw_text: str = Field(min_length=1)
    expected: Expected | None

    @field_validator("expected", mode="before")
    @classmethod
    def empty_means_unannotated(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("use {} for a case that is not annotated yet")
        if value == {}:
            return None
        return value

    @property
    def annotated(self) -> bool:
        return self.expected is not None


class DatasetError(ValueError):
    """The dataset file is malformed. The message names the line."""


def load_dataset(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: dict[str, int] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = EvalCase.model_validate(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{number}: invalid JSON: {exc}") from exc
        except ValidationError as exc:
            raise DatasetError(f"{path}:{number}: {exc}") from exc
        if case.id in seen:
            raise DatasetError(
                f"{path}:{number}: duplicate id {case.id!r} "
                f"(first seen on line {seen[case.id]})"
            )
        seen[case.id] = number
        cases.append(case)
    if not cases:
        raise DatasetError(f"{path}: no cases")
    return cases
