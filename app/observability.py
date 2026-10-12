"""
Observability = Logs + Metrics + Traces.

- LOGS:    structured JSON (one line per event) -> shipped to ELK / Loki.
- METRICS: Prometheus counters/histograms scraped at /metrics -> Grafana.
- TRACES:  every request gets a trace_id; each pipeline step records a span
           (name, duration, attributes). Spans are returned in the API response
           and logged, so you can see exactly where latency goes.
           (Drop-in upgrade path: OpenTelemetry / LangSmith / Langfuse.)

DISCUSSION TALKING POINT: "I can answer 'why was this answer slow / wrong /
expensive?' for any single request, and see p95 latency + cost trends overall."
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any

from prometheus_client import Counter, Histogram


# ---------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": current_trace_id(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)  # type: ignore[attr-defined]
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def log_event(logger: logging.Logger, msg: str, **fields: Any) -> None:
    logger.info(msg, extra={"extra_fields": fields})


# ---------------------------------------------------------------- metrics
REQUESTS = Counter("pp_requests_total", "RAG requests", ["route", "status"])
LATENCY = Histogram(
    "pp_step_latency_seconds", "Latency per pipeline step", ["step"],
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
TOKENS = Counter("pp_llm_tokens_total", "LLM tokens", ["direction"])
COST = Counter("pp_llm_cost_usd_total", "Estimated LLM spend in USD")
CACHE = Counter("pp_cache_total", "Cache lookups", ["result"])
GUARDRAIL = Counter("pp_guardrail_total", "Guardrail triggers", ["kind"])
HITL = Counter("pp_hitl_total", "Answers routed to human review")
FEEDBACK = Counter("pp_feedback_total", "User feedback", ["rating"])

# ---------------------------------------------------------------- tracing
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_spans: contextvars.ContextVar[list | None] = contextvars.ContextVar("spans", default=None)


def current_trace_id() -> str:
    return _trace_id.get()


def start_trace() -> str:
    tid = uuid.uuid4().hex[:16]
    _trace_id.set(tid)
    _spans.set([])
    return tid


def get_spans() -> list[dict]:
    return list(_spans.get() or [])


@contextmanager
def span(name: str, **attrs: Any):
    """Time a block, record it as a span AND as a Prometheus histogram sample."""
    t0 = time.perf_counter()
    record: dict[str, Any] = {"name": name, **attrs}
    try:
        yield record          # caller may add attributes: record["hits"] = 5
    finally:
        dt = time.perf_counter() - t0
        record["ms"] = round(dt * 1000, 2)
        LATENCY.labels(step=name).observe(dt)
        spans = _spans.get()
        if spans is not None:
            spans.append(record)
