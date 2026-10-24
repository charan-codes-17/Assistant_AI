"""add_conversations_and_chat_messages

Revision ID: 3f7c1b9e52ad
Revises: 1acdf692cf1a
Create Date: 2026-10-24 13:05:00.000000

Server-side chat history: two additive tables. No existing table is altered.
Both foreign keys use ON DELETE CASCADE so a /forget (delete_user_account) or the admin
probe script, which both end with a plain DELETE on the user row, remove the history too.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '3f7c1b9e52ad'
down_revision: Union[str, None] = '1acdf692cf1a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'conversations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=120), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_conversations_user_id_updated_at',
        'conversations',
        ['user_id', sa.text('updated_at DESC')],
        unique=False,
    )
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("role IN ('user', 'assistant')", name='ck_chat_messages_role'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_chat_messages_conversation_id_created_at',
        'chat_messages',
        ['conversation_id', 'created_at'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_chat_messages_conversation_id_created_at', table_name='chat_messages')
    op.drop_table('chat_messages')
    op.drop_index('ix_conversations_user_id_updated_at', table_name='conversations')
    op.drop_table('conversations')
