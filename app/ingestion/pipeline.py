"""
The ETL pipeline:  Extract -> Transform (clean, chunk) -> Load (vector DB + SQL).

Properties that make it production-grade:
  * IDEMPOTENT  - each doc's content hash is stored in SQL; re-running the
                  pipeline skips unchanged docs (no duplicate vectors, no wasted
                  embedding cost). Changed docs are re-chunked & replaced.
  * OBSERVABLE  - returns a run report (counts, timings, data-quality stats).
  * DECOUPLED   - the same `ingest_documents()` is called by the CLI, the API,
                  a Celery worker, and the Kafka streaming consumer.

CLI:
    python -m app.ingestion.pipeline --source sample
    python -m app.ingestion.pipeline --source arxiv --query "cat:cs.CL AND abs:retrieval" --max 50
    python -m app.ingestion.pipeline --source dir --path ./my_pdfs
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from app.config import get_settings
from app.db import Document, get_session
from app.ingestion.chunking import chunk_document
from app.ingestion.cleaning import clean_documents
from app.ingestion.loaders import RawDocument, fetch_arxiv, load_directory, load_file
from app.retrieval import refresh_keyword_index
from app.vectorstore import get_store

logger = logging.getLogger(__name__)
SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample" / "papers.json"


def ingest_documents(docs: list[RawDocument]) -> dict:
    s = get_settings() # return base settings for safety, guard rails, cost, queueing like kafka
    
    t0 = time.perf_counter() # performance counter for benchmark

    #Clean + dedupe + redact. Returns docs and a small data-quality report
    docs, quality = clean_documents(docs, redact=s.redact_pii) 

    # get session of sqlite
    session = get_session()

    # get chromadb or pinecone store
    store = get_store()

    report = {"quality": quality, "new": 0, "updated": 0, "skipped": 0, "chunks": 0}

    for doc in docs:
        existing = session.get(Document, doc.doc_id)
        if existing and existing.content_hash == doc.content_hash:
            report["skipped"] += 1
            continue
        if existing:
            store.delete_doc(doc.doc_id)          # content changed -> replace its chunks
            report["updated"] += 1
        else:
            report["new"] += 1

        chunks = chunk_document(doc, s.chunk_size, s.chunk_overlap, s.chunk_strategy)

        store.upsert(chunks)

        report["chunks"] += len(chunks)

        session.merge(Document(
            doc_id=doc.doc_id, title=doc.title, source=doc.source, url=doc.url,
            published=doc.published, content_hash=doc.content_hash, n_chunks=len(chunks),
        ))

    session.commit()
    session.close()
    refresh_keyword_index()
    report["seconds"] = round(time.perf_counter() - t0, 2)
    report["total_vectors"] = store.count()
    logger.info("ingestion done %s", report)
    
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="PaperPilot ingestion pipeline")
    p.add_argument("--source", choices=["sample", "arxiv", "dir", "file"], default="sample")
    p.add_argument("--query", default="cat:cs.CL AND abs:retrieval augmented generation")
    p.add_argument("--max", type=int, default=25)
    p.add_argument("--path", default="")
    a = p.parse_args()

    if a.source == "sample":
        docs = load_file(SAMPLE)
    elif a.source == "arxiv":
        docs = fetch_arxiv(a.query, a.max)
    elif a.source == "dir":
        docs = load_directory(a.path)
    else:
        docs = load_file(Path(a.path))
    print(json.dumps(ingest_documents(docs), indent=2))


if __name__ == "__main__":
    main()
