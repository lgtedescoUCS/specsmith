"""Repository against a real PostgreSQL. Skipped when it is not running."""

from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from specsmith.db import repository
from specsmith.db.models import (
    ConstraintItemRow,
    OpenQuestionRow,
    RequirementRow,
    SpecRow,
)
from specsmith.domain.errors import SpecNotFoundError
from specsmith.domain.models import Spec

SPEC = Spec.model_validate(
    {
        "objective": "Increase tempering line throughput",
        "current_state": {"throughput": "400 plates/day", "furnace": "2007"},
        "target_state": {"throughput": "650 plates/day"},
        "constraints": [
            {"kind": "space", "value": "12 m of available floor space"},
            {"kind": "budget", "value": "approx. USD 1,200,000"},
            {"kind": "deadline", "value": "operational by March"},
        ],
        "open_questions": [
            "What glass thickness range must the line handle?",
            "How many shifts per day?",
            "Is the existing furnace replaced or kept in parallel?",
        ],
        "confidence": "medium",
    }
)


def count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_saved_spec_round_trips_unchanged(db_session: Session) -> None:
    saved = repository.save_spec(
        db_session, raw_text="raw input", source="unit-test", spec=SPEC
    )
    db_session.expunge_all()  # force a real read instead of the identity map

    fetched = repository.get_spec(db_session, saved.id)

    assert fetched == saved
    identity = {"id", "requirement_id", "created_at"}
    assert fetched.model_dump(exclude=identity) == SPEC.model_dump()


def test_list_order_is_preserved(db_session: Session) -> None:
    spec = SPEC.model_copy(
        update={"open_questions": ["Zeta?", "Alpha?", "Mu?"]}
    )
    saved = repository.save_spec(db_session, raw_text="x", source=None, spec=spec)
    db_session.expunge_all()

    fetched = repository.get_spec(db_session, saved.id)

    assert fetched.open_questions == ["Zeta?", "Alpha?", "Mu?"]
    assert [c.kind for c in fetched.constraints] == ["space", "budget", "deadline"]


def test_minimal_spec_round_trips_with_empty_collections(db_session: Session) -> None:
    spec = Spec.model_validate({"objective": "Reduce scrap", "confidence": "low"})
    saved = repository.save_spec(db_session, raw_text="x", source=None, spec=spec)
    db_session.expunge_all()

    fetched = repository.get_spec(db_session, saved.id)

    assert fetched.current_state == {}
    assert fetched.target_state == {}
    assert fetched.constraints == []
    assert fetched.open_questions == []


def test_save_stores_requirement_text_and_server_timestamp(db_session: Session) -> None:
    saved = repository.save_spec(
        db_session, raw_text="raw input", source="email", spec=SPEC
    )

    requirement = db_session.get(RequirementRow, saved.requirement_id)

    assert requirement is not None
    assert requirement.raw_text == "raw input"
    assert requirement.source == "email"
    assert saved.created_at.tzinfo is not None


def test_get_unknown_spec_raises_not_found(db_session: Session) -> None:
    missing = uuid4()

    with pytest.raises(SpecNotFoundError) as exc_info:
        repository.get_spec(db_session, missing)

    assert exc_info.value.spec_id == missing


def test_deleting_requirement_cascades_to_spec_and_children(
    db_session: Session,
) -> None:
    saved = repository.save_spec(db_session, raw_text="x", source=None, spec=SPEC)
    db_session.expunge_all()

    # Delete at the SQL level so the database's ON DELETE CASCADE is what
    # gets tested, not SQLAlchemy's in-memory cascade.
    db_session.execute(
        RequirementRow.__table__.delete().where(
            RequirementRow.id == saved.requirement_id
        )
    )

    assert count(db_session, SpecRow) == 0
    assert count(db_session, ConstraintItemRow) == 0
    assert count(db_session, OpenQuestionRow) == 0
