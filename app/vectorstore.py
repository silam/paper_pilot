"""
Vector Database layer.

`VectorStore` is an interface; `ChromaStore` is the default implementation
(embedded on disk locally, client/server in Docker). `PineconeStore` shows the
managed-cloud alternative behind the same interface - the rest of the code
never imports chromadb or pinecone directly. (Weaviate would be a third class.)

Chroma vs Pinecone vs Weaviate (know this for the DISCUSSION):
  * Chroma   - open-source, embedded or server, great for dev & small/medium prod.
  * Pinecone - fully managed, serverless, scales to billions; you pay per usage.
  * Weaviate - open-source, built-in hybrid (BM25+vector) search, GraphQL API.
  All use ANN indexes (usually HNSW): approximate search trading a little
  recall for log-time queries. Tunables: M, ef_construction, ef_search.

We pass our OWN embeddings (instead of letting Chroma embed) so the embedding
model is a single, versioned decision in our code.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.embeddings import get_embedder
from app.ingestion.chunking import Chunk

from rich import print

@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    score: float           # higher = more relevant (cosine similarity)


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> int: ...
    def query(self, text: str, k: int, where: dict | None = None) -> list[Hit]: ...
    def all_chunks(self) -> list[Hit]: ...
    def delete_doc(self, doc_id: str) -> None: ...
    def count(self) -> int: ...


class ChromaStore:
    def __init__(self):
        import chromadb
        s = get_settings()
        self.embedder = get_embedder()
        if s.chroma_mode == "http":
            self.client = chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)
        else:
            self.client = chromadb.PersistentClient(path=s.chroma_path)
        self.col = self.client.get_or_create_collection(
            name=s.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedder.name},
        )

    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        batch = 128
        for i in range(0, len(chunks), batch):
            b = chunks[i:i + batch]
            self.col.upsert(
                ids=[c.chunk_id for c in b],
                documents=[c.text for c in b],
                embeddings=self.embedder.embed([c.text for c in b]),
                metadatas=[{k: v for k, v in c.metadata.items() if v not in (None, "")} for c in b],
            )
        return len(chunks)

    def query(self, text: str, k: int, where: dict | None = None) -> list[Hit]:

        print(f"Chrome Query = {text}")
        if self.col.count() == 0:
            return []
        res = self.col.query(query_embeddings=self.embedder.embed([text]),
                             n_results=min(k, self.col.count()), where=where)
        
        print(f"############# Res={res}")
        return [
            
            Hit(chunk_id=i, text=d, metadata=m or {}, score=1.0 - dist)   # cosine distance -> similarity

            for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0])
        ]

    def all_chunks(self) -> list[Hit]:
        res = self.col.get(include=["documents", "metadatas"])
        return [Hit(chunk_id=i, text=d, metadata=m or {}, score=0.0)
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def delete_doc(self, doc_id: str) -> None:
        """GDPR 'right to be forgotten' - remove every chunk of a document."""
        self.col.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self.col.count()


class PineconeStore:  # pragma: no cover - needs a cloud account
    """Same interface, managed cloud backend. Enable with VECTOR_BACKEND=pinecone."""

    def __init__(self):
        from pinecone import Pinecone
        s = get_settings()
        self.embedder = get_embedder()
        self.index = Pinecone(api_key=s.pinecone_api_key).Index(s.pinecone_index)
        self._text: dict[str, str] = {}

    def upsert(self, chunks: list[Chunk]) -> int:
        vecs = self.embedder.embed([c.text for c in chunks])
        self.index.upsert(vectors=[
            {"id": c.chunk_id, "values": v, "metadata": {**c.metadata, "text": c.text}}
            for c, v in zip(chunks, vecs)
        ])
        return len(chunks)

    def query(self, text: str, k: int, where: dict | None = None) -> list[Hit]:

        print(f"PineCone{text}")
        res = self.index.query(vector=self.embedder.embed([text])[0], top_k=k,
                               include_metadata=True, filter=where)
        return [Hit(m.id, m.metadata.get("text", ""), dict(m.metadata), m.score) for m in res.matches]

    def all_chunks(self) -> list[Hit]:
        return []   # BM25 side would use a separate keyword index in this setup

    def delete_doc(self, doc_id: str) -> None:
        self.index.delete(filter={"doc_id": doc_id})

    def count(self) -> int:
        return self.index.describe_index_stats().total_vector_count


@lru_cache
def get_store() -> VectorStore:
    return PineconeStore() if get_settings().vector_backend == "pinecone" else ChromaStore()
