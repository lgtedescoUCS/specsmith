"""create extraction_run table

Revision ID: 47d918efe329
Revises: c21c6a0856b7
Create Date: 2026-11-06 11:13:30.584098

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '47d918efe329'
down_revision: Union[str, Sequence[str], None] = 'c21c6a0856b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('extraction_run',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('requirement_id', sa.Uuid(), nullable=False),
    sa.Column('spec_id', sa.Uuid(), nullable=True),
    sa.Column('model', sa.String(length=80), nullable=False),
    sa.Column('prompt_version', sa.String(length=20), nullable=False),
    sa.Column('input_tokens', sa.Integer(), nullable=False),
    sa.Column('output_tokens', sa.Integer(), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=False),
    sa.Column('attempts', sa.SmallInteger(), server_default=sa.text('1'), nullable=False),
    sa.Column('succeeded', sa.Boolean(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('input_tokens >= 0 AND output_tokens >= 0 AND latency_ms >= 0 AND attempts >= 1', name=op.f('ck_extraction_run_metrics_in_range')),
    sa.CheckConstraint('succeeded = (error IS NULL)', name=op.f('ck_extraction_run_error_set_iff_failed')),
    sa.ForeignKeyConstraint(['requirement_id'], ['requirement.id'], name=op.f('fk_extraction_run_requirement_id_requirement'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['spec_id'], ['spec.id'], name=op.f('fk_extraction_run_spec_id_spec'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_extraction_run')),
    sa.UniqueConstraint('spec_id', name=op.f('uq_extraction_run_spec_id'))
    )
    op.create_index(op.f('ix_extraction_run_requirement_id'), 'extraction_run', ['requirement_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_extraction_run_requirement_id'), table_name='extraction_run')
    op.drop_table('extraction_run')
