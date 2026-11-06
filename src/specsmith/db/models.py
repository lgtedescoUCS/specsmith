"""SQLAlchemy models. This module knows nothing about the LLM.

Row classes carry a `Row` suffix so they never get confused with the
Pydantic domain models of the same name.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    MetaData,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Deterministic constraint names, so Alembic migrations can refer to them.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class RequirementRow(Base):
    __tablename__ = "requirement"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    raw_text: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SpecRow(Base):
    __tablename__ = "spec"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    requirement_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("requirement.id", ondelete="CASCADE"), index=True
    )
    objective: Mapped[str] = mapped_column(Text)
    current_state: Mapped[dict[str, str]] = mapped_column(
        JSONB, server_default=text("'{}'::jsonb")
    )
    target_state: Mapped[dict[str, str]] = mapped_column(
        JSONB, server_default=text("'{}'::jsonb")
    )
    confidence: Mapped[str] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    requirement: Mapped[RequirementRow] = relationship()
    # Children are ordered by `position`: a normalised table has no order of
    # its own, and the lists must come back exactly as they were stored.
    constraints: Mapped[list["ConstraintItemRow"]] = relationship(
        order_by="ConstraintItemRow.position",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    open_questions: Mapped[list["OpenQuestionRow"]] = relationship(
        order_by="OpenQuestionRow.position",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )


class ConstraintItemRow(Base):
    __tablename__ = "constraint_item"
    # The unique (spec_id, position) pair also serves as the index on spec_id.
    __table_args__ = (UniqueConstraint("spec_id", "position"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    spec_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("spec.id", ondelete="CASCADE")
    )
    position: Mapped[int] = mapped_column(SmallInteger)
    kind: Mapped[str] = mapped_column(String(20))
    value: Mapped[str] = mapped_column(Text)


class OpenQuestionRow(Base):
    __tablename__ = "open_question"
    __table_args__ = (UniqueConstraint("spec_id", "position"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    spec_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("spec.id", ondelete="CASCADE")
    )
    position: Mapped[int] = mapped_column(SmallInteger)
    text: Mapped[str] = mapped_column(Text)
