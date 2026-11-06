"""The shape the LLM is asked to produce, and its conversion to a Spec.

It differs from the domain Spec in one place: current_state and target_state
are lists of {name, value} pairs instead of dicts. Structured output requires
`additionalProperties: false` on every object, which rules out a free-form
dict; a list of pairs is the closest shape it accepts.

The JSON schema built here is vendor-neutral and keeps every constraint
(lengths, item counts). A provider whose API ignores or rejects some of them
strips them itself; Pydantic validation here enforces all of them anyway.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from specsmith.domain.models import (
    MAX_OPEN_QUESTIONS,
    OBJECTIVE_MAX_LENGTH,
    Confidence,
    Constraint,
    OpenQuestion,
    Spec,
)


class _DraftModel(BaseModel):
    # Same rules as the domain models: unknown keys are errors, and
    # whitespace-only strings do not count as values.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StateEntry(_DraftModel):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)


class SpecDraft(_DraftModel):
    # Every field is required, with no defaults: the model must state empty
    # lists explicitly instead of leaving a field out.
    objective: str = Field(min_length=1, max_length=OBJECTIVE_MAX_LENGTH)
    current_state: list[StateEntry]
    target_state: list[StateEntry]
    constraints: list[Constraint]
    open_questions: list[OpenQuestion] = Field(max_length=MAX_OPEN_QUESTIONS)
    confidence: Confidence

    @field_validator("current_state", "target_state")
    @classmethod
    def names_are_unique(cls, entries: list[StateEntry]) -> list[StateEntry]:
        # Converting to a dict would silently keep only the last value of a
        # repeated name. Rejecting it sends the error back to the model.
        # Names are compared after whitespace stripping, case-sensitively,
        # exactly as they would collide as dict keys.
        seen: set[str] = set()
        duplicates: list[str] = []
        for entry in entries:
            if entry.name in seen and entry.name not in duplicates:
                duplicates.append(entry.name)
            seen.add(entry.name)
        if duplicates:
            names = ", ".join(repr(name) for name in duplicates)
            raise ValueError(f"duplicate name {names}; each name must appear once")
        return entries

    def to_spec(self) -> Spec:
        """Build the domain Spec, applying its rules (lengths, item counts).

        Raises pydantic.ValidationError when a domain rule fails. Field
        names are the same in both models, so error locations read the same.
        """
        return Spec.model_validate(
            {
                "objective": self.objective,
                "current_state": {e.name: e.value for e in self.current_state},
                "target_state": {e.name: e.value for e in self.target_state},
                "constraints": [c.model_dump() for c in self.constraints],
                "open_questions": self.open_questions,
                "confidence": self.confidence,
            }
        )


def parse_spec(content: str) -> Spec:
    """Raw LLM output -> Spec. Raises pydantic.ValidationError on any problem,
    from invalid JSON to a domain rule, so all of them take the retry path."""
    return SpecDraft.model_validate_json(content).to_spec()


SPEC_DRAFT_SCHEMA: dict[str, Any] = SpecDraft.model_json_schema()
