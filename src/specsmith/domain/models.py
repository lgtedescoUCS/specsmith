"""Domain models for a technical specification.

This module knows nothing about the database or the LLM. The same models
validate API input today and LLM output from Phase 2 on.
"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ConstraintKind(StrEnum):
    BUDGET = "budget"
    DEADLINE = "deadline"
    SPACE = "space"
    THROUGHPUT = "throughput"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"
    OTHER = "other"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class _DomainModel(BaseModel):
    # Unknown fields are rejected rather than silently dropped, so an LLM
    # inventing extra keys fails validation instead of passing unnoticed.
    # Whitespace is stripped so that "   " cannot satisfy min_length=1.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Constraint(_DomainModel):
    kind: ConstraintKind
    value: str = Field(min_length=1, max_length=300)


OpenQuestion = Annotated[str, Field(min_length=1)]


class Spec(_DomainModel):
    objective: str = Field(min_length=1, max_length=300)
    current_state: dict[str, str] = Field(default_factory=dict)
    target_state: dict[str, str] = Field(default_factory=dict)
    constraints: list[Constraint] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list, max_length=10)
    confidence: Confidence


class StoredSpec(Spec):
    """A spec that has been persisted, as returned by the API."""

    id: UUID
    requirement_id: UUID
    created_at: datetime
