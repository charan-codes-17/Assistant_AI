"""add_user_skills

Revision ID: b91e4d7a2c30
Revises: 3f7c1b9e52ad
Create Date: 2026-10-25 10:00:00.000000

User-defined skills: one additive table. No existing table is altered.
The foreign key uses ON DELETE CASCADE so a /forget (delete_user_account) or the admin probe
script, which both end with a plain DELETE on the user row, remove the user's skills too.
Built-in skills are a Python registry and never produce rows here.

The instructions column is Text; the 1500 character cap is enforced in the API, not in the
schema, to keep this migration purely additive and free of CHECK constraints.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b91e4d7a2c30'
down_revision: Union[str, None] = '3f7c1b9e52ad'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'user_skills',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=40), nullable=False),
        sa.Column('description', sa.String(length=200), server_default='', nullable=False),
        sa.Column('instructions', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'name', name='uq_user_skills_user_id_name'),
    )
    op.create_index('ix_user_skills_user_id', 'user_skills', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_user_skills_user_id', table_name='user_skills')
    op.drop_table('user_skills')
