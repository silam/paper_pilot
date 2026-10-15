"""
The RAG pipeline as a LangGraph state machine (Corrective-RAG + Self-check).

                 ┌──────────────┐ blocked
   question ───▶ │ input_guard  │────────────────────────────────▶ END
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │ rewrite      │◀──────────────┐ (no relevant docs,
                 └──────┬───────┘               │  retry once with the
                        ▼                       │  original wording / more BM25)
                 ┌──────────────┐               │
                 │ retrieve     │  hybrid search + feature-store boost
                 └──────┬───────┘               │
                        ▼                       │
                 ┌──────────────┐  none relevant│
                 │ grade_docs   │───────────────┘
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │ generate     │  versioned prompt, sanitised context
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐ low confidence ┌──────────────┐
                 │ validate     │───────────────▶│ human_review │──▶ END
                 └──────┬───────┘                └──────────────┘
                        ▼ ok
                       END

Why a graph instead of a linear chain?
  * Conditional edges (retry, escalate) are explicit and testable.
  * Each node is a pure function of state -> easy unit tests & tracing.
  * Adding a node (e.g. a reranker, a tool call, a web-search fallback) doesn't
    require rewriting the pipeline.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.feature_store import feature_boost, get_feature_store
from app.guardrails import (
    _content_words,
    check_citations,
    check_input,
    check_output,
    groundedness,
    sanitize_context,
)
from app.llm.client import get_llm
from app.llm.prompts import QUERY_REWRITE, format_context, get_prompt
from app.observability import GUARDRAIL, HITL, span
from app.retrieval import hybrid_search
from app.vectorstore import Hit


class RAGState(TypedDict, total=False):
    question: str
    prompt_version: str
    filters: dict | None
    # working fields
    query: str
    attempts: int
    alpha: float | None
    hits: list[Hit]
    relevant: list[Hit]
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    # verdicts
    status: str                     # answered | blocked | needs_review
    guard_reasons: list[str]
    groundedness: dict
    citations: dict


# ------------------------------------------------------------------ nodes
def input_guard(state: RAGState) -> RAGState:
    with span("input_guard") as sp:
        if not get_settings().enable_guardrails:
            return {"query": state["question"], "guard_reasons": []}
        g = check_input(state["question"])
        sp["reasons"] = g.reasons
        for r in g.reasons:
            GUARDRAIL.labels(r).inc()
        if not g.allowed:
            return {"status": "blocked", "guard_reasons": g.reasons,
                    "answer": "Request blocked by safety guardrails."}

        print(f"********* Reason = {g.reasons}")
        return {"question": g.sanitized, "query": g.sanitized, "guard_reasons": g.reasons}


def rewrite(state: RAGState) -> RAGState:
    attempts = state.get("attempts", 0)
    with span("rewrite", attempt=attempts):
        if attempts > 0:
            # Corrective retry: fall back to the raw question & lean on keywords.
            return {"query": state["question"], "alpha": 0.3, "attempts": attempts + 1}
        llm = get_llm()
        
        if llm.provider == "offline":
            print(f"********* llm.provider = {llm.provider}")
            return {"query": state["question"], "attempts": 1}

        
        system, user = QUERY_REWRITE.render(question=state["question"])
        r = llm.complete(system, user, max_tokens=60)

        print(f"********* text query = {r.text.strip()}")
        return {"query": r.text.strip() or state["question"], "attempts": 1,
                **_usage(state, r)}


def retrieve(state: RAGState) -> RAGState:
    with span("retrieve") as sp:
        hits = hybrid_search(state["query"], where=state.get("filters"), alpha=state.get("alpha"))
        # Online feature lookup -> tiny boost for docs users rated well.
        feats = get_feature_store().get_doc_features([h.metadata.get("doc_id", "") for h in hits])
        for h in hits:
            h.score += feature_boost(feats.get(h.metadata.get("doc_id", ""), {}))
        hits.sort(key=lambda h: h.score, reverse=True)
        sp["hits"] = len(hits)
        return {"hits": hits}


def grade_docs(state: RAGState) -> RAGState:
    """
    Lightweight retrieval evaluator (the CRAG idea): drop chunks that aren't
    about the question BEFORE they reach the LLM. Irrelevant context is the
    main cause of RAG hallucinations, and it also wastes tokens.

    relevance = max(semantic similarity, keyword coverage of the question)
    (Production upgrade: a cross-encoder or a small LLM grader returning yes/no.)
    """
    # with ... as sp:: This is Python's context manager syntax. 
    # It guarantees that a specific setup happens before the code runs 
    # and a cleanup happens after it finishes—even if your code crashes or raises an error.
    # span("grade_docs"): This initializes a "span"—a single unit of work 
    # or an execution segment (in this case, the process of grading documents).
    # as sp: This binds the span object to the variable sp. Inside the block, 
    # you can use sp to log metrics, attach metadata 
    # (like sp.set_attribute("num_docs", 5)), or track exceptions.

    with span("grade_docs") as sp:
        thr = get_settings().min_relevance
        qwords = _content_words(state["question"])
        relevant = []
        for h in state["hits"]:
            coverage = len(qwords & _content_words(h.text)) / len(qwords) if qwords else 0.0
            h.metadata["relevance"] = round(max(h.score, coverage), 3)
            if h.metadata["relevance"] >= thr:
                relevant.append(h)
        sp["relevant"] = len(relevant)
        return {"relevant": relevant}

#-------------------------------------------------------------------
def generate(state: RAGState) -> RAGState:
    with span("generate") as sp:
        relevant = state.get("relevant", [])
        if not relevant:
            return {"answer": "I don't know based on the indexed papers.", "model": "none"}
        safe_hits = []
        for h in relevant:
            text, flagged = sanitize_context(h.text)
            if flagged:
                GUARDRAIL.labels("indirect_injection_in_context").inc()
            safe_hits.append(Hit(h.chunk_id, text, h.metadata, h.score))

        prompt = get_prompt(state.get("prompt_version", "rag_v1"))
        system, user = prompt.render(context=format_context(safe_hits), question=state["question"])
        
        r = get_llm().complete(system, user)
        sp.update(model=r.model, in_tok=r.input_tokens, out_tok=r.output_tokens)
        return {"answer": r.text.strip(), "model": r.model, "relevant": safe_hits, **_usage(state, r)}


def validate(state: RAGState) -> RAGState:
    with span("validate") as sp:
        answer, leaks = check_output(state["answer"]) #edacts PII and returns any detected leaks. 
        sources = [h.text for h in state.get("relevant", [])]

        print(f"********* answer = {answer}")
        print(f"********* sources = {sources}")
        print(f"********* leaks = {leaks}")
        
        g = groundedness(answer, sources) # compares the answer’s sentences with the relevant source texts and returns a score plus sentence-level details.

        c = check_citations(answer, len(sources)) # finds citation numbers such as [1] and flags numbers outside the range 1 to len(sources).

        print(f"********* groundedness = {g}")
        print(f"********* citations = {c}")
        
        sp.update(groundedness=g["score"], invalid_citations=c["invalid"])

        for leak in leaks:
            GUARDRAIL.labels(leak).inc()

        return {"answer": answer, "groundedness": g, "citations": c,
                "guard_reasons": state.get("guard_reasons", []) + leaks}

#----------------------------------------------------------------------
def human_review(state: RAGState) -> RAGState:
    HITL.inc()
    with span("human_review"):
        return {"status": "needs_review"}

#----------------------------------------------------------------------
def finalize(state: RAGState) -> RAGState:
    return {"status": "answered"}


#---------------------------------------------------------------------- routing
def after_guard(state: RAGState) -> str:
    return "blocked" if state.get("status") == "blocked" else "ok"


def after_grade(state: RAGState) -> str:
    if not state.get("relevant") and state.get("attempts", 0) < 2:
        return "retry"
    return "generate"


def after_validate(state: RAGState) -> str:
    s = get_settings()
    g = state.get("groundedness", {}).get("score", 0.0)
    bad_cites = state.get("citations", {}).get("invalid")
    no_cites = not state.get("citations", {}).get("has_citations", True)
    if g < s.hitl_confidence_threshold or bad_cites or no_cites:
        return "review"
    return "ok"


def _usage(state: RAGState, r) -> dict[str, Any]:
    return {"input_tokens": state.get("input_tokens", 0) + r.input_tokens,
            "output_tokens": state.get("output_tokens", 0) + r.output_tokens,
            "cost_usd": state.get("cost_usd", 0.0) + r.cost_usd}


# ------------------------------------------------------------------ build
def build_graph():
    g = StateGraph(RAGState)
    for name, fn in [("input_guard", input_guard), ("rewrite", rewrite), ("retrieve", retrieve),
                     ("grade_docs", grade_docs), ("generate", generate), ("validate", validate),
                     ("human_review", human_review), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", after_guard, {"blocked": END, "ok": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", after_grade, {"retry": "rewrite", "generate": "generate"})
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", after_validate, {"review": "human_review", "ok": "finalize"})
    g.add_edge("human_review", END)
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def run_rag(question: str, prompt_version: str = "rag_v1", filters: dict | None = None) -> RAGState:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH.invoke({"question": question, "prompt_version": prompt_version,
                          "filters": filters, "attempts": 0})
