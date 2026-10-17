"""
backend/models/models.py
SQLAlchemy 2.0 async models for AI Assistant per roadmap §3.5 and Phase 1 task 1.1.
All timestamps are timezone-aware. Foreign keys from user-owned tables use ON DELETE CASCADE.
Vector dimension is dynamically loaded from settings.EMBEDDING_DIM.
"""

from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.config.settings import settings
from backend.models.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[Optional[str]] = mapped_column(String(100), unique=True, nullable=True, index=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    telegram_chat_id: Mapped[Optional[int]] = mapped_column(BigInteger, unique=True, nullable=True, index=True)
    timezone: Mapped[str] = mapped_column(String(50), nullable=False, default="Asia/Kolkata", server_default="Asia/Kolkata")
    quiet_mode: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=sa.false())
    onboarding_state: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default="not_started", server_default="not_started")
    last_active_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships with CASCADE delete
    link_codes: Mapped[List["LinkCode"]] = relationship(back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    sessions: Mapped[List["ConversationSession"]] = relationship(back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    knowledge_records: Mapped[List["KnowledgeRecord"]] = relationship(back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    goals: Mapped[List["Goal"]] = relationship(back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    projects: Mapped[List["Project"]] = relationship(back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    tasks: Mapped[List["Task"]] = relationship(back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    profile_summary: Mapped[Optional["ProfileSummary"]] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan", passive_deletes=True)


class LinkCode(Base):
    __tablename__ = "link_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="link_codes")


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)  # "telegram" | "web"
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_active_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="sessions")
    inspector_events: Mapped[List["InspectorEvent"]] = relationship(back_populates="session", cascade="all, delete-orphan", passive_deletes=True)


class KnowledgeRecord(Base):
    __tablename__ = "knowledge_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)  # "knowledge" | "decision" | "preference" | "glossary" | "ephemeral"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    certainty: Mapped[str] = mapped_column(String(20), nullable=False, default="idea", server_default="idea")  # "idea" | "tentative" | "decided"
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="text", server_default="text")  # "text" | "voice" | "import" | "questionnaire"
    source_ref: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(settings.EMBEDDING_DIM), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", server_default="pending")  # "pending" | "confirmed" | "rejected"
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_by: Mapped[Optional[int]] = mapped_column(ForeignKey("knowledge_records.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped["User"] = relationship(back_populates="knowledge_records")
    superseded_record: Mapped[Optional["KnowledgeRecord"]] = relationship(remote_side=[id])

    __table_args__ = (
        Index(
            "ix_knowledge_records_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Goal(Base):
    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active", server_default="active")
    target_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="goals")
    projects: Mapped[List["Project"]] = relationship(back_populates="goal")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    goal_id: Mapped[Optional[int]] = mapped_column(ForeignKey("goals.id", ondelete="SET NULL"), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active", server_default="active")

    user: Mapped["User"] = relationship(back_populates="projects")
    goal: Mapped[Optional["Goal"]] = relationship(back_populates="projects")
    milestones: Mapped[List["Milestone"]] = relationship(back_populates="project", cascade="all, delete-orphan", passive_deletes=True)
    tasks: Mapped[List["Task"]] = relationship(back_populates="project")
    progress_notes: Mapped[List["Progress"]] = relationship(back_populates="project", cascade="all, delete-orphan", passive_deletes=True)


class Milestone(Base):
    __tablename__ = "milestones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    due_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", server_default="pending")

    project: Mapped["Project"] = relationship(back_populates="milestones")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id: Mapped[Optional[int]] = mapped_column(ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    due_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open", server_default="open")  # "open" | "done" | "snoozed"
    origin: Mapped[str] = mapped_column(String(20), nullable=False, default="manual", server_default="manual")  # "manual" | "extracted" | "promise"
    snoozed_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_reminded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="tasks")
    project: Mapped[Optional["Project"]] = relationship(back_populates="tasks")


class Progress(Base):
    __tablename__ = "progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    note: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    project: Mapped["Project"] = relationship(back_populates="progress_notes")


class ProfileSummary(Base):
    __tablename__ = "profile_summaries"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    blob: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user: Mapped["User"] = relationship(back_populates="profile_summary")


class InspectorEvent(Base):
    __tablename__ = "inspector_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("conversation_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    turn_id: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieved_record_ids: Mapped[List[int]] = mapped_column(ARRAY(Integer), nullable=False, default=list, server_default="{}")
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    directive: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped["ConversationSession"] = relationship(back_populates="inspector_events")

