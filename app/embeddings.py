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

#----------------------------------------------------------------------------------
class Embedder(Protocol):
    name: str
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...

#----------------------------------------------------------------------------------
    
def _normalize(v: list[float]) -> list[float]:
    print("*********** _normalize")

    # Compute the L2 norm of the vector. If the norm is zero, use 1.0 to avoid division by zero.
    # Return the normalized vector.
    # This ensures that the resulting vector has a unit length.
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedder:
    """Deterministic hashing-trick embedder (offline fallback)."""

    def __init__(self, dim: int = 384):
        print("*********** HashEmbedder.__init__")
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> list[str]:
        print("*********** HashEmbedder._features")
        # Extract tokens from the text using the predefined regular expression.
        # Also generate bigram features by combining consecutive tokens.   
        #  Lower-case the text and take every word and every pair of adjacent words (bigram) as a “feature”.
        # example: "hello world" -> ["hello", "world", "hello_world"]
        # example: "Hello, Si Lam, good morning, how are you?" ->
        # ["hello", "si", "lam", "good", "morning", "how", "are", "you",
        #  "hello_si", "si_lam", "lam_good", "good_morning", "morning_how",
        #  "how_are", "are_you"]
        toks = _TOKEN.findall(text.lower())

        # Return the list of tokens combined with bigram features.
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def embed(self, texts: list[str]) -> list[list[float]]:
        print("*********** HashEmbedder.embed")
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for f in self._features(t):
                # Compute a hash for the feature and update the corresponding dimension in the vector.
                # Use the MD5 hash of the feature to determine its position and the sign in the embedding vector.
                # The hash value is used to determine both the index in the embedding vector and the sign of the update.
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)

                # Determine the index in the embedding vector using the hash modulo the dimension.
                # Determine the sign of the update using the 9th bit of the hash.   
                """
                # The index in the embedding vector is determined by the hash modulo the dimension.
                # The sign of the update is determined by the 9th bit of the hash.
                This line updates one position in the vector v for the current text feature. 
                The index is h % self.dim, so the feature’s hash h maps it to one of the vector’s self.dim positions.
                Multiple features can map to the same position; each one adds to or subtracts from that position.

                The conditional chooses the sign of the update: (h >> 8) & 1 extracts the hash’s bit at position 8. If that bit is 1, the code adds 1.0; if it is 0, it adds -1.0. 
                This technique is called signed hashing: it lets a compact vector represent many features while partly reducing the bias caused by hash collisions.

                
                """
                
                v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0   # signed hashing
            out.append(_normalize(v))
        return out


class HFEmbedder:
    """Hugging Face sentence-transformers embedder."""

    def __init__(self, model_name: str):
        print("*********** HFEmbedder.__init__")
        from sentence_transformers import SentenceTransformer  # heavy import -> lazy
        self.model = SentenceTransformer(model_name)
        #self.dim = self.model.get_sentence_embedding_dimension()
        self.dim = self.model.get_embedding_dimension()
        self.name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        print("*********** HFEmbedder.embed")
        vecs = self.model.encode(texts, batch_size=32, 
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
        return vecs.tolist()


@lru_cache
def get_embedder() -> Embedder:
    print("*********** get_embedder")
    s = get_settings()
    if s.embedding_backend == "hf":
        return HFEmbedder(s.embedding_model)
    return HashEmbedder(s.embedding_dim)


def cosine(a: list[float], b: list[float]) -> float:
    print("*********** cosine")
    return sum(x * y for x, y in zip(a, b))   # inputs are normalised
