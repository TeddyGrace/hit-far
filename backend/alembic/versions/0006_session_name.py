"""session name

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-24 13:00:00

Lets the golfer give a range session a name, editable alongside location and notes on the
session page. Existing rows keep name NULL; the session page falls back to the date.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0006'
down_revision: str | None = '0005'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('sessions', sa.Column('name', sa.String(200)))


def downgrade() -> None:
    op.drop_column('sessions', 'name')
