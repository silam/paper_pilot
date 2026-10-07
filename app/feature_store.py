"""
Feature Store (offline -> online), the minimal version you can explain on a whiteboard.

A feature store gives you the SAME feature values at training/batch time
(offline store, e.g. Postgres/BigQuery/Parquet) and at serving time (online
store, e.g. Redis), avoiding "training-serving skew".

Here the features describe DOCUMENTS and are learned from user behaviour:
  * doc_citation_count  - how often the doc was cited in answers
  * doc_feedback_score  - mean thumbs-up/down on answers citing the doc
  * doc_freshness_days  - age of the paper

Flow:
  materialize()  (batch job, run by Celery beat / cron)
      SQL query_logs + feedback  ->  compute features  ->  write to online store
  get_doc_features(doc_ids)  (online, at request time, ~1ms)
      used by the RAG graph to gently boost well-liked, recent papers.

feature_repo/ contains the equivalent Feast definitions for the production path.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import select

from app.config import get_settings
from app.db import Document, Feedback, QueryLog, get_session

DEFAULTS = {"doc_citation_count": 0, "doc_feedback_score": 0.0, "doc_freshness_days": 9999}


class FeatureStore:
    def __init__(self):
        s = get_settings()
        self.redis = None
        if s.redis_url:
            import redis
            self.redis = redis.Redis.from_url(s.redis_url, decode_responses=True)
        self._mem: dict[str, dict] = {}

    def _write(self, doc_id: str, feats: dict) -> None:
        if self.redis:
            self.redis.hset("pp:features:doc", doc_id, json.dumps(feats))
        else:
            self._mem[doc_id] = feats

    def get_doc_features(self, doc_ids: list[str]) -> dict[str, dict]:
        if self.redis and doc_ids:
            raw = self.redis.hmget("pp:features:doc", doc_ids)
            return {d: (json.loads(r) if r else dict(DEFAULTS)) for d, r in zip(doc_ids, raw)}
        return {d: self._mem.get(d, dict(DEFAULTS)) for d in doc_ids}

    def materialize(self) -> int:
        """Offline computation -> online store. Returns #docs updated."""
        session = get_session()
        citations: dict[str, int] = defaultdict(int)
        ratings: dict[str, list[int]] = defaultdict(list)
        fb = {f.query_id: f.rating for f in session.scalars(select(Feedback))}
        for q in session.scalars(select(QueryLog)):
            for src in q.sources or []:
                citations[src] += 1
                if q.id in fb:
                    ratings[src].append(fb[q.id])
        now = datetime.now(UTC)
        n = 0
        for doc in session.scalars(select(Document)):
            try:
                pub = datetime.fromisoformat(doc.published.replace("Z", "+00:00"))
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=UTC)
                age = (now - pub).days
            except ValueError:
                age = DEFAULTS["doc_freshness_days"]
            r = ratings.get(doc.doc_id, [])
            self._write(doc.doc_id, {
                "doc_citation_count": citations.get(doc.doc_id, 0),
                "doc_feedback_score": round(sum(r) / len(r), 3) if r else 0.0,
                "doc_freshness_days": age,
            })
            n += 1
        session.close()
        return n


def feature_boost(feats: dict) -> float:
    """Small, bounded boost so behaviour signals break ties, never dominate relevance."""
    return 0.03 * max(-1.0, min(1.0, feats.get("doc_feedback_score", 0.0)))


@lru_cache
def get_feature_store() -> FeatureStore:
    return FeatureStore()
