"""Feast definitions mirroring app/feature_store.py (doc-level behaviour features).

    cd feature_repo && feast apply && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

document = Entity(name="doc_id", join_keys=["doc_id"])

doc_stats_source = FileSource(
    path="../data/features/doc_stats.parquet",     # written by a batch job from query_logs
    timestamp_field="event_timestamp",
)

doc_stats = FeatureView(
    name="doc_stats",
    entities=[document],
    ttl=timedelta(days=7),
    schema=[
        Field(name="doc_citation_count", dtype=Int64),
        Field(name="doc_feedback_score", dtype=Float32),
        Field(name="doc_freshness_days", dtype=Int64),
    ],
    source=doc_stats_source,
)
