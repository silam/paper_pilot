"""
Caching (Redis) - the #1 cost & latency optimisation for LLM apps.

Two layers:
  1. EXACT cache    key = sha256(normalised question + prompt version + filters)
                    O(1) lookup, zero risk.
  2. SEMANTIC cache store (question embedding -> answer). A new question whose
                    embedding has cosine >= threshold (e.g. 0.95) with a cached
                    one reuses that answer: "What is LoRA?" ~ "Explain LoRA".
                    Higher hit-rate, small risk of a wrong reuse -> keep the
                    threshold high and include the prompt version in the key.

Backend: Redis if REDIS_URL is set (shared across API replicas), otherwise an
in-process dict (fine for local dev). TTL avoids serving stale answers after
re-ingestion. (Redis 8 / Redis Stack also offer native vector search - an
upgrade path for a large semantic cache.)
"""
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache

from app.config import get_settings
from app.embeddings import cosine, get_embedder
from app.observability import CACHE


def _norm(q: str) -> str:
    print("*********** _norm")
    return " ".join(q.lower().split())


class AnswerCache:
    def __init__(self):
        print("*********** AnswerCache.__init__")
        s = get_settings()
        self.ttl = s.cache_ttl_seconds
        self.threshold = s.semantic_cache_threshold
        self.redis = None
        if s.redis_url:
            import redis
            self.redis = redis.Redis.from_url(s.redis_url, decode_responses=True)
        self._mem: dict[str, tuple[float, str]] = {}
        self._semantic: list[tuple[list[float], str, str]] = []   # (vec, namespace, key)

    # ---- low-level kv
    def _get(self, key: str) -> str | None:
        print("*********** AnswerCache._get")
        if self.redis:
            return self.redis.get(key)
        item = self._mem.get(key)
        if item and item[0] > time.time():
            return item[1]
        return None

    def _set(self, key: str, value: str) -> None:
        print("*********** AnswerCache._set")
        if self.redis:
            self.redis.setex(key, self.ttl, value)
        else:
            self._mem[key] = (time.time() + self.ttl, value)

    # ---- public API
    def key(self, question: str, namespace: str) -> str:
        print("*********** AnswerCache.key")
        return "pp:ans:" + hashlib.sha256(f"{namespace}|{_norm(question)}".encode()).hexdigest()

    def get(self, question: str, namespace: str) -> tuple[dict | None, str]:
        print("*********** AnswerCache.get")
        hit = self._get(self.key(question, namespace))
        if hit:
            CACHE.labels("exact_hit").inc()
            return json.loads(hit), "exact"
        qv = get_embedder().embed([_norm(question)])[0]
        best, best_key = 0.0, None
        for vec, ns, k in self._load_semantic():
            if ns == namespace:
                sim = cosine(qv, vec)
                if sim > best:
                    best, best_key = sim, k
        if best_key and best >= self.threshold:
            value = self._get(best_key)
            if value:
                CACHE.labels("semantic_hit").inc()
                return json.loads(value), "semantic"
        CACHE.labels("miss").inc()
        return None, "miss"

    def set(self, question: str, namespace: str, value: dict) -> None:
        print("*********** AnswerCache.set")
        k = self.key(question, namespace)
        self._set(k, json.dumps(value))
        vec = get_embedder().embed([_norm(question)])[0]
        if self.redis:
            self.redis.rpush("pp:semantic", json.dumps([vec, namespace, k]))
            self.redis.ltrim("pp:semantic", -5000, -1)          # bound memory
        else:
            self._semantic.append((vec, namespace, k))
            self._semantic = self._semantic[-5000:]

    def _load_semantic(self):
        print("*********** AnswerCache._load_semantic")
        if self.redis:
            return [tuple(json.loads(x)) for x in self.redis.lrange("pp:semantic", 0, -1)]
        return self._semantic

    def clear(self) -> None:
        print("*********** AnswerCache.clear")
        if self.redis:
            for k in self.redis.scan_iter("pp:*"):
                self.redis.delete(k)
        self._mem.clear()
        self._semantic.clear()


@lru_cache
def get_cache() -> AnswerCache:
    print("*********** get_cache")
    return AnswerCache()
