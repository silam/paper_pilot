"""
Async work with Celery (task queue) on a Redis broker (RabbitMQ also works:
set CELERY_BROKER_URL=amqp://...).

Why a queue?  Ingesting 500 papers (download + chunk + embed) takes minutes.
The API must answer in milliseconds, so it ENQUEUES the job and returns a
task_id. Workers scale horizontally and independently of the API
(that's the microservice split: api pods vs worker pods).

Celery beat runs scheduled jobs: nightly arXiv refresh + feature materialisation.

Run a worker:  celery -A app.workers.tasks worker -B --loglevel=info
"""
from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

s = get_settings()
celery_app = Celery("paperpilot", broker=s.celery_broker_url, backend=s.celery_broker_url)
celery_app.conf.update(
    task_acks_late=True,                 # re-deliver if a worker dies mid-task
    worker_prefetch_multiplier=1,        # fair dispatch for long tasks
    task_time_limit=1800,
    beat_schedule={
        "nightly-arxiv-refresh": {
            "task": "app.workers.tasks.ingest_arxiv_task",
            "schedule": crontab(hour=2, minute=0),
            "args": ("cat:cs.CL AND abs:retrieval augmented generation", 50),
        },
        "hourly-feature-materialize": {
            "task": "app.workers.tasks.materialize_features_task",
            "schedule": crontab(minute=5),
        },
    },
)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def ingest_arxiv_task(self, query: str, max_results: int = 25) -> dict:
    from app.ingestion.loaders import fetch_arxiv
    from app.ingestion.pipeline import ingest_documents
    return ingest_documents(fetch_arxiv(query, max_results))


@celery_app.task
def materialize_features_task() -> int:
    from app.feature_store import get_feature_store
    return get_feature_store().materialize()
