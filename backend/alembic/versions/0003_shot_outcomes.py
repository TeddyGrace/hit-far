"""shot outcomes

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-23 18:00:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0003'
down_revision: str | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'shot_outcomes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('swing_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('swings.id', ondelete='CASCADE'),
                  nullable=False, unique=True),
        sa.Column('shape', sa.String(10)),
        sa.Column('start_line', sa.String(10)),
        sa.Column('contact', sa.String(10)),
        sa.Column('source', sa.String(20), nullable=False, server_default='self'),
        sa.Column('club_path', sa.Float()),
        sa.Column('face_to_path', sa.Float()),
        sa.Column('face_angle', sa.Float()),
        sa.Column('carry', sa.Float()),
        sa.Column('offline', sa.Float()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('shot_outcomes')
