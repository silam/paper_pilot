"""
Hybrid Retrieval = dense (vector) + sparse (BM25 keyword) + fusion + rerank.

Why hybrid? Dense vectors capture meaning ("car" ~ "automobile") but miss exact
tokens (model names, acronyms like "LoRA", arXiv IDs). BM25 nails exact terms
but misses paraphrases. Combining them is the industry default.

Fusion: Reciprocal Rank Fusion (RRF)
    score(d) = sum over rankers r of  w_r / (k + rank_r(d))
RRF only uses ranks, so we don't need to calibrate BM25 scores against cosine
scores (they're on totally different scales).

Optional rerank: a cross-encoder (query, chunk) -> relevance score. Slower but
much more precise; applied only to the top ~20 fused candidates.

DISCUSSION TALKING POINT: "Retrieval quality is the ceiling on answer quality.
I measure it separately (hit-rate / MRR on a golden set) before touching prompts."
"""
from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.vectorstore import Hit, get_store

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in and or for on with is are was were be by as at from that this it its".split())


def tokenize(text: str) -> list[str]:
    print("*********** tokenize")
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP] # if t is not in the set of STOP words


class BM25Index:
    """In-process keyword index rebuilt when the corpus changes."""

    def __init__(self):
        print("*********** BM25Index.__init__")
        self._lock = threading.Lock()
        self._version = -1
        self._bm25: BM25Okapi | None = None
        self._hits: list[Hit] = []

    def refresh(self, force: bool = False) -> None:
        print("*********** BM25Index.refresh")
        store = get_store()
        version = store.count()          # cheap change detector
        with self._lock:
            if not force and version == self._version:
                return
            self._hits = store.all_chunks()
            corpus = [tokenize(h.text) for h in self._hits] or [[""]]
            self._bm25 = BM25Okapi(corpus)
            self._version = version

    def search(self, query: str, k: int) -> list[Hit]:

        print("*********** BM25Index.search")
        print(f"Search = {query}")

        self.refresh()
        if not self._hits or self._bm25 is None:
            return []
        
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._hits, scores), key=lambda x: x[1], reverse=True)[:k]
        return [Hit(h.chunk_id, h.text, h.metadata, float(s)) for h, s in ranked if s > 0]


_bm25 = BM25Index()

#---------------------------------------------------------------------------------------
# score(chunk) = sum(weight_i / (k + rank_i(chunk) + 1))
"""
This function is doing Reciprocal Rank Fusion (RRF), 
which combines multiple ranked result lists into one ordered list.

What it does:

ranked_lists: a list of result lists, e.g. dense vector hits and BM25 hits
weights: how much to trust each list
k: the RRF constant, usually around 60
Core idea:

For each hit in each list, compute: w / (k + rank + 1)
Add that to the hit’s fused score
After all lists are processed, sort all unique chunk IDs by 
the fused score descending
Return the hits in that order
Why this matters:

RRF is rank-based, not score-based.
It avoids needing to normalize cosine similarity and BM25 scores, which are on different scales.
Important details in your code:

fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + ...
Accumulates the total fused score for each chunk
by_id.setdefault(h.chunk_id, h)
Keeps the first copy of a hit for that chunk
This is intentional: the comment says the dense list should win, 
so the first list’s Hit object is preserved
order = sorted(fused, key=fused.get, reverse=True)
Sorts chunk IDs by fused score, highest first
return [by_id[i] for i in order]
Returns the actual Hit objects in fused order
A subtle gotcha:

zip(ranked_lists, weights) silently truncates if lengths differ.
If the lists are not same-length, you may miss data without an error.
A safer version would validate lengths before looping.
Also:

The print(...) calls are debug noise and should be removed in production.
They are not part of the actual ranking logic.
The function is conceptually:


and then sort by that score.

"""

def rrf_fuse(ranked_lists: list[list[Hit]], weights: list[float], k: int = 60) -> list[Hit]:
    print("*********** rrf_fuse")
    fused: dict[str, float] = {}
    by_id: dict[str, Hit] = {}
    for hits, w in zip(ranked_lists, weights):
        for rank, h in enumerate(hits):
            print(f"########## rank = {rank}") # 0,1,2,4
            print(f"########## h = {h}") # Hit(chunk_id='3783d28ea75d3a3e', text='

            # Accumulates the total fused score for each chunk
            fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + w / (k + rank + 1)

            print(f"############# fused[h.chunk_id] = {fused[h.chunk_id]}")
            # first list = dense, so its cosine score wins (used by the grader)
            by_id.setdefault(h.chunk_id, h)

    # Sorts chunk IDs by fused score, highest first
    order = sorted(fused, key=fused.get, reverse=True)

    #Returns the actual Hit objects in fused order
    return [by_id[i] for i in order]

#---------------------------------------------------------------------------------------
class CrossEncoderReranker:  # optional, enabled when sentence-transformers is installed
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print("*********** CrossEncoderReranker.__init__")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        print("*********** CrossEncoderReranker.rerank")
        scores = self.model.predict([(query, h.text) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:k]
        return [h for h, _ in ranked]

#----------------------------------------------------------------------------------
def hybrid_search(query: str, k: int | None = None, where: dict | None = None,
                  alpha: float | None = None) -> list[Hit]:
    print("*********** hybrid_search")
    s = get_settings()
    k = k or s.top_k
    alpha = s.hybrid_alpha if alpha is None else alpha
    candidates = k * 4                        # over-fetch, then fuse/trim

    print(f"hybrid_search query = {query}, where = {where}")

    dense = get_store().query(query, candidates, where=where)
    print(f"############## dense = {dense}")
    sparse = _bm25.search(query, candidates) if alpha < 1.0 else []
    print(f"################ sparse = {sparse}")
    fused = rrf_fuse([dense, sparse], [alpha, 1 - alpha])
    print(f"################ fused = {fused}")   

    # Make sure every fused hit has a cosine score (BM25-only hits come in
    # with a BM25 score) so downstream relevance grading is consistent.
    from app.embeddings import cosine, get_embedder
    emb = get_embedder()
    qv = emb.embed([query])[0]
    dense_ids = {h.chunk_id for h in dense}
    missing = [h for h in fused[:k] if h.chunk_id not in dense_ids]
    if missing:
        for h, v in zip(missing, emb.embed([m.text for m in missing])):
            h.score = cosine(qv, v)
    return fused[:k]


def refresh_keyword_index() -> None:
    print("*********** refresh_keyword_index")
    _bm25.refresh(force=True)
