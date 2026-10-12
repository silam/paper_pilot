"""
Embeddings & Vector Representations.

An embedding maps text -> a dense vector where semantic similarity becomes
geometric closeness (cosine similarity). Retrieval = nearest-neighbour search.

Two backends behind ONE interface (Strategy pattern):
  * HFEmbedder   - Hugging Face sentence-transformers (all-MiniLM-L6-v2, 384-d).
                   Real semantic model; runs on CPU; no API cost.
  * HashEmbedder - "feature hashing" of word unigrams+bigrams into a fixed-size
                   vector. NOT semantic, but deterministic, instant, zero deps.
                   Used in unit tests / CI so the pipeline is testable offline.

Vectors are L2-normalised, so dot product == cosine similarity.

DISCUSSION TALKING POINTS:
  - "Embedding model choice is an eval-driven decision (MTEB leaderboard is a
     starting point, your own golden set is the decider)."
  - "Batching + caching embeddings by content hash saves cost on re-ingestion."
  - "Same model must be used for docs and queries - changing it means re-indexing,
     so the model name is stored in collection metadata."
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from app.config import get_settings

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedder:
    """Deterministic hashing-trick embedder (offline fallback)."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> list[str]:
        toks = _TOKEN.findall(text.lower())
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for f in self._features(t):
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0   # signed hashing
            out.append(_normalize(v))
        return out


class HFEmbedder:
    """Hugging Face sentence-transformers embedder."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import -> lazy
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False)
        return vecs.tolist()


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_backend == "hf":
        return HFEmbedder(s.embedding_model)
    return HashEmbedder(s.embedding_dim)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))   # inputs are normalised
