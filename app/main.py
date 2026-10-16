"""
FastAPI service - the "front door" microservice.

Endpoints
  POST /ask                 question -> grounded, cited answer (+ trace, cost, confidence)
  POST /ingest              trigger ETL (sync, or async via Celery queue)
  POST /upload              upload a PDF/MD/TXT and index it
  POST /feedback            thumbs up/down -> continuous-improvement loop
  GET  /review              Human-in-the-Loop queue of low-confidence answers
  POST /review/{id}         reviewer approves / corrects an answer
  DELETE /documents/{id}    GDPR erasure: remove a doc from SQL + vector index
  GET  /stats/ab            A/B report per prompt version (SQL analytics)
  POST /features/materialize  batch-compute features -> online store
  GET  /health, /ready      Kubernetes liveness / readiness probes
  GET  /metrics             Prometheus scrape endpoint

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs  (auto-generated OpenAPI / Swagger UI)
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.ab_testing import assign_variant
from app.cache import get_cache
from app.config import get_settings
from app.db import Document, Feedback, QueryLog, ReviewItem, get_session, prompt_version_report
from app.feature_store import get_feature_store
from app.ingestion.loaders import fetch_arxiv, load_file
from app.ingestion.pipeline import SAMPLE, ingest_documents
from app.observability import FEEDBACK, REQUESTS, get_spans, log_event, setup_logging, start_trace
from app.rag_graph import run_rag
from app.retrieval import refresh_keyword_index
from app.vectorstore import get_store

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("paperpilot.api")

app = FastAPI(title="PaperPilot", version="1.0.0",
              description="Production-style RAG assistant for AI research papers")


# ------------------------------------------------------------------ schemas
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    user_id: str = "anonymous"
    prompt_version: str | None = None          # None -> A/B assignment
    filters: dict | None = None                # e.g. {"source": "arxiv"}
    use_cache: bool = True


class Source(BaseModel):
    n: int
    doc_id: str
    title: str
    url: str = ""
    score: float
    snippet: str


class AskResponse(BaseModel):
    query_id: int | None
    answer: str
    status: str
    sources: list[Source]
    groundedness: float
    prompt_version: str
    model: str
    cache: str
    latency_ms: float
    cost_usd: float
    trace_id: str
    trace: list[dict]
    guard_reasons: list[str] = []


class IngestRequest(BaseModel):
    source: str = "sample"                     # sample | arxiv
    query: str = "cat:cs.CL AND abs:retrieval augmented generation"
    max_results: int = 20
    run_async: bool = False


class FeedbackRequest(BaseModel):
    query_id: int
    rating: int = Field(..., ge=-1, le=1)
    comment: str = ""


class ReviewAction(BaseModel):
    action: str = Field(..., pattern="^(approve|correct)$")
    answer: str = ""


# ------------------------------------------------------------------ middleware
_buckets: dict[str, deque] = defaultdict(deque)
RATE_LIMIT = 30          # requests per minute per client (abuse & cost protection)


@app.middleware("http")
async def rate_limit_and_trace(request: Request, call_next):
    print("*********** rate_limit_and_trace")
    if request.url.path == "/ask":
        ip = request.client.host if request.client else "unknown"
        q, now = _buckets[ip], time.time()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            REQUESTS.labels("/ask", "429").inc()
            return Response("rate limit exceeded", status_code=429)
        q.append(now)
    return await call_next(request)


# ------------------------------------------------------------------ core endpoint
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    print("*********** ask")
    trace_id = start_trace()
    t0 = time.perf_counter()
    version = req.prompt_version or assign_variant(req.user_id)
    namespace = f"{version}|{sorted((req.filters or {}).items())}"
    cache = get_cache()

    cached, cache_status = (cache.get(req.question, namespace) if req.use_cache else (None, "bypass"))
    if cached:
        state = cached
    else:
        print(f"{req.question}")
        result = run_rag(req.question, version, req.filters)
        relevant = result.get("relevant", [])
        state = {
            "answer": result.get("answer", ""),
            "status": result.get("status", "answered"),
            "groundedness": result.get("groundedness", {}).get("score", 0.0),
            "model": result.get("model", "none"),
            "cost_usd": round(result.get("cost_usd", 0.0), 6),
            "guard_reasons": result.get("guard_reasons", []),
            "sources": [
                {"n": i, "doc_id": h.metadata.get("doc_id", ""), "title": h.metadata.get("title", ""),
                 "url": h.metadata.get("url", ""), "score": round(h.score, 3),
                 "snippet": h.text[:300]}
                for i, h in enumerate(relevant, 1)
            ],
        }
        # Only cache confident, answered results - never cache blocked/needs_review.
        if state["status"] == "answered" and req.use_cache:
            cache.set(req.question, namespace, state)

    latency = round((time.perf_counter() - t0) * 1000, 1)

    # Persist to SQL (query log + HITL queue).
    session = get_session()
    log = QueryLog(trace_id=trace_id, user_id=req.user_id, question=req.question,
                   answer=state["answer"], prompt_version=version, model=state["model"],
                   groundedness=state["groundedness"], latency_ms=latency,
                   cost_usd=0.0 if cached else state["cost_usd"], cache=cache_status,
                   status=state["status"], sources=sorted({s["doc_id"] for s in state["sources"]}))
    session.add(log)
    session.flush()
    if state["status"] == "needs_review":
        session.add(ReviewItem(query_id=log.id, reason=f"groundedness={state['groundedness']}"))
    session.commit()
    query_id = log.id
    session.close()

    REQUESTS.labels("/ask", state["status"]).inc()
    log_event(logger, "ask", query_id=query_id, status=state["status"], latency_ms=latency,
              groundedness=state["groundedness"], cache=cache_status, prompt_version=version)
    return AskResponse(query_id=query_id, prompt_version=version, cache=cache_status,
                       latency_ms=latency, trace_id=trace_id, trace=get_spans(),
                       **{k: state[k] for k in ("answer", "status", "sources", "groundedness",
                                                 "model", "cost_usd", "guard_reasons")})


# ------------------------------------------------------------------ ingestion
@app.post("/ingest")
def ingest(req: IngestRequest):
    print("*********** ingest")
    if req.run_async:
        from app.workers.tasks import ingest_arxiv_task
        task = ingest_arxiv_task.delay(req.query, req.max_results)
        return {"task_id": task.id, "status": "queued"}
    docs = load_file(SAMPLE) if req.source == "sample" else fetch_arxiv(req.query, req.max_results)
    report = ingest_documents(docs)
    get_cache().clear()          # corpus changed -> cached answers may be stale
    return report


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    print("*********** upload")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".md", ".txt"}:
        raise HTTPException(400, "only .pdf, .md, .txt supported")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(file.filename).name
        with path.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        report = ingest_documents(load_file(path))
    get_cache().clear()
    return report


@app.delete("/documents/{doc_id:path}")
def delete_document(doc_id: str):
    """GDPR Art. 17 erasure: vectors + catalogue + cached answers."""
    print("*********** delete_document")
    get_store().delete_doc(doc_id)
    session = get_session()
    doc = session.get(Document, doc_id)
    if doc:
        session.delete(doc)
        session.commit()
    session.close()
    refresh_keyword_index()
    get_cache().clear()
    return {"deleted": doc_id}


@app.get("/documents")
def list_documents():
    print("*********** list_documents")
    session = get_session()
    docs = session.scalars(select(Document)).all()
    session.close()
    return [{"doc_id": d.doc_id, "title": d.title, "source": d.source, "chunks": d.n_chunks} for d in docs]


# ------------------------------------------------------------------ feedback & HITL
@app.post("/feedback")
def feedback(req: FeedbackRequest):
    print("*********** feedback")
    session = get_session()
    if not session.get(QueryLog, req.query_id):
        raise HTTPException(404, "unknown query_id")
    session.add(Feedback(query_id=req.query_id, rating=req.rating, comment=req.comment))
    if req.rating < 0:   # thumbs-down -> also goes to human review
        session.add(ReviewItem(query_id=req.query_id, reason="user_thumbs_down"))
    session.commit()
    session.close()
    FEEDBACK.labels("up" if req.rating > 0 else "down").inc()
    return {"ok": True}


@app.get("/review")
def review_queue(status: str = "pending"):
    print("*********** review_queue")
    session = get_session()
    rows = session.execute(
        select(ReviewItem, QueryLog).join(QueryLog, QueryLog.id == ReviewItem.query_id)
        .where(ReviewItem.status == status)
    ).all()
    session.close()
    return [{"review_id": r.id, "query_id": q.id, "reason": r.reason, "question": q.question,
             "answer": q.answer, "groundedness": q.groundedness} for r, q in rows]


@app.post("/review/{review_id}")
def review_decide(review_id: int, action: ReviewAction):
    print("*********** review_decide")
    session = get_session()
    item = session.get(ReviewItem, review_id)
    if not item:
        raise HTTPException(404, "unknown review item")
    item.status = "approved" if action.action == "approve" else "corrected"
    item.reviewer_answer = action.answer
    session.commit()
    session.close()
    return {"review_id": review_id, "status": item.status}


# ------------------------------------------------------------------ analytics & ops
@app.get("/stats/ab")
def ab_stats():
    print("*********** ab_stats")
    session = get_session()
    report = prompt_version_report(session)
    session.close()
    return report


@app.post("/features/materialize")
def materialize_features():
    print("*********** materialize_features")
    return {"docs_updated": get_feature_store().materialize()}


@app.get("/health")
def health():
    print("*********** health")
    return {"status": "ok"}


@app.get("/ready")
def ready():
    print("*********** ready")
    try:
        n = get_store().count()
        get_session().close()
    except Exception as exc:
        raise HTTPException(503, f"not ready: {exc}") from exc
    return {"status": "ready", "vectors": n, "llm_provider": settings.llm_provider,
            "embedding_backend": settings.embedding_backend}


@app.get("/metrics")
def metrics():
    print("*********** metrics")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
