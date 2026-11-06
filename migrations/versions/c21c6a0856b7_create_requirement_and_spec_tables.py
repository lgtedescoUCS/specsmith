"""create requirement and spec tables

Revision ID: c21c6a0856b7
Revises: 
Create Date: 2026-11-06 08:25:49.318954

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c21c6a0856b7'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('requirement',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('raw_text', sa.Text(), nullable=False),
    sa.Column('source', sa.String(length=80), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_requirement'))
    )
    op.create_table('spec',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('requirement_id', sa.Uuid(), nullable=False),
    sa.Column('objective', sa.Text(), nullable=False),
    sa.Column('current_state', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('target_state', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('confidence', sa.String(length=10), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['requirement_id'], ['requirement.id'], name=op.f('fk_spec_requirement_id_requirement'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_spec'))
    )
    op.create_index(op.f('ix_spec_requirement_id'), 'spec', ['requirement_id'], unique=False)
    op.create_table('constraint_item',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('spec_id', sa.Uuid(), nullable=False),
    sa.Column('position', sa.SmallInteger(), nullable=False),
    sa.Column('kind', sa.String(length=20), nullable=False),
    sa.Column('value', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['spec_id'], ['spec.id'], name=op.f('fk_constraint_item_spec_id_spec'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_constraint_item')),
    sa.UniqueConstraint('spec_id', 'position', name=op.f('uq_constraint_item_spec_id_position'))
    )
    op.create_table('open_question',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('spec_id', sa.Uuid(), nullable=False),
    sa.Column('position', sa.SmallInteger(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['spec_id'], ['spec.id'], name=op.f('fk_open_question_spec_id_spec'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_open_question')),
    sa.UniqueConstraint('spec_id', 'position', name=op.f('uq_open_question_spec_id_position'))
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('open_question')
    op.drop_table('constraint_item')
    op.drop_index(op.f('ix_spec_requirement_id'), table_name='spec')
    op.drop_table('spec')
    op.drop_table('requirement')
