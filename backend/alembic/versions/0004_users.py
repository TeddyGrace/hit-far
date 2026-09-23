"""users

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-23 20:00:00

Existing rows keep user_id NULL here; API start-up creates the first user (from APP_PASSWORD) and
assigns them, since a migration can't know the password.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0004'
down_revision: str | None = '0003'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('username', sa.String(50), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(200), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.add_column('sessions', sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id')))
    op.create_index('ix_sessions_user_id', 'sessions', ['user_id'])
    for table in ('models', 'datasets'):
        op.add_column(table, sa.Column('user_id', postgresql.UUID(as_uuid=True),
                                       sa.ForeignKey('users.id', ondelete='CASCADE')))
        op.create_index(f'ix_{table}_user_id', table, ['user_id'])


def downgrade() -> None:
    for table in ('datasets', 'models', 'sessions'):
        op.drop_index(f'ix_{table}_user_id', table)
        op.drop_column(table, 'user_id')
    op.drop_table('users')
