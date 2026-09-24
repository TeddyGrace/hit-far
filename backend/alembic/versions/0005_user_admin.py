"""user admin flag

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-24 12:00:00

The first user (the owner) becomes the admin, so they can add users from the web app.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0005'
down_revision: str | None = '0004'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('users', sa.Column('is_admin', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.execute("UPDATE users SET is_admin = true WHERE id = (SELECT id FROM users ORDER BY created_at LIMIT 1)")


def downgrade() -> None:
    op.drop_column('users', 'is_admin')
