"""
Relational store (SQL) - the system of record for everything that ISN'T a vector.

SQL vs NoSQL in this project:
  * SQL (Postgres / SQLite)  - documents catalogue, query logs, feedback,
                               human-review queue. Relational, needs joins
                               and analytics ("thumbs-up rate per prompt version").
  * NoSQL key-value (Redis)  - cache + online feature store: low-latency lookups.
  * Vector DB (Chroma)       - similarity search over embeddings.
  Picking the right store per access pattern is a classic system-design question.

The QueryLog + Feedback tables are the backbone of the CONTINUOUS IMPROVEMENT
LOOP: bad answers (low groundedness / thumbs-down) become new eval cases.
"""
from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    doc_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    source: Mapped[str] = mapped_column(String(50))
    url: Mapped[str] = mapped_column(String(500), default="")
    published: Mapped[str] = mapped_column(String(50), default="")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class QueryLog(Base):
    __tablename__ = "query_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)
    user_id: Mapped[str] = mapped_column(String(100), default="anonymous")
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(100))
    groundedness: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cache: Mapped[str] = mapped_column(String(20), default="miss")
    status: Mapped[str] = mapped_column(String(30), default="answered")  # answered|blocked|needs_review
    sources: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("query_logs.id"))
    rating: Mapped[int] = mapped_column(Integer)          # +1 / -1
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ReviewItem(Base):
    """Human-in-the-Loop queue: low-confidence answers wait for a reviewer."""
    __tablename__ = "review_queue"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("query_logs.id"))
    reason: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="pending")   # pending|approved|corrected
    reviewer_answer: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


@lru_cache
def _engine():
    url = get_settings().database_url
    if url.startswith("sqlite:///"):
        from pathlib import Path
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(engine)       # demo-friendly; use Alembic migrations in prod
    return engine


def get_session() -> Session:
    return sessionmaker(bind=_engine(), expire_on_commit=False)()


def prompt_version_report(session: Session) -> list[dict]:
    """SQL analytics for A/B testing: quality, latency, cost & thumbs per variant."""
    rows = session.execute(
        select(
            QueryLog.prompt_version,
            func.count(QueryLog.id),
            func.avg(QueryLog.groundedness),
            func.avg(QueryLog.latency_ms),
            func.sum(QueryLog.cost_usd),
            func.avg(Feedback.rating),
        ).outerjoin(Feedback, Feedback.query_id == QueryLog.id)
         .where(QueryLog.status != "blocked")        # blocked requests never reached the prompt
         .group_by(QueryLog.prompt_version)
    ).all()
    return [
        {"prompt_version": r[0], "queries": r[1], "avg_groundedness": round(r[2] or 0, 3),
         "avg_latency_ms": round(r[3] or 0, 1), "total_cost_usd": round(r[4] or 0, 5),
         "avg_feedback": None if r[5] is None else round(r[5], 2)}
        for r in rows
    ]
