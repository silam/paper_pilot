"""
Streaming ingestion with Kafka (same pattern applies to AWS Kinesis).

Batch ETL runs nightly; streaming makes new documents searchable within
seconds of publication.

   producer (arXiv poller / any upstream system)
        │  JSON paper events
        ▼
   Kafka topic  "papers.raw"   (partitioned, durable, replayable)
        │
        ▼
   consumer group "paperpilot-indexer"  (scale = #partitions)
        │  micro-batches of N events or T seconds
        ▼
   ingest_documents()  ->  clean -> chunk -> embed -> vector DB

Key ideas to mention:
  * Consumer groups give horizontal scaling + fault tolerance.
  * Commit offsets AFTER successful indexing -> at-least-once delivery;
    idempotent ingestion (content-hash) makes duplicates harmless.
  * Micro-batching amortises embedding calls (much cheaper than 1-by-1).
  * Bad messages go to a dead-letter topic instead of blocking the partition.

Run:
  python -m app.streaming.kafka_pipeline produce --query "abs:LLM agents" --max 10
  python -m app.streaming.kafka_pipeline consume
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from app.config import get_settings

logger = logging.getLogger(__name__)


def produce(query: str, max_results: int) -> int:
    print("*********** produce")
    from kafka import KafkaProducer

    from app.ingestion.loaders import fetch_arxiv
    s = get_settings()
    producer = KafkaProducer(bootstrap_servers=s.kafka_bootstrap,
                             value_serializer=lambda v: json.dumps(v).encode(),
                             key_serializer=lambda k: k.encode())
    docs = fetch_arxiv(query, max_results)
    for d in docs:
        producer.send(s.kafka_topic, key=d.doc_id, value=d.__dict__)   # key -> same partition per doc
    producer.flush()
    return len(docs)


def consume(batch_size: int = 16, max_wait_s: float = 5.0) -> None:
    print("*********** consume")
    from kafka import KafkaConsumer, KafkaProducer

    from app.ingestion.loaders import RawDocument
    from app.ingestion.pipeline import ingest_documents
    s = get_settings()
    consumer = KafkaConsumer(
        s.kafka_topic, bootstrap_servers=s.kafka_bootstrap, group_id="paperpilot-indexer",
        enable_auto_commit=False, auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode()),
    )
    dlq = KafkaProducer(bootstrap_servers=s.kafka_bootstrap,
                        value_serializer=lambda v: json.dumps(v).encode())
    batch, started = [], time.time()
    for msg in consumer:
        try:
            batch.append(RawDocument(**msg.value))
        except TypeError as exc:
            dlq.send(s.kafka_topic + ".dlq", {"error": str(exc), "payload": msg.value})
        if len(batch) >= batch_size or time.time() - started > max_wait_s:
            if batch:
                report = ingest_documents(batch)
                logger.info("indexed micro-batch %s", report)
            consumer.commit()                 # commit only after success
            batch, started = [], time.time()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["produce", "consume"])
    p.add_argument("--query", default="abs:retrieval augmented generation")
    p.add_argument("--max", type=int, default=10)
    a = p.parse_args()
    if a.mode == "produce":
        print(f"produced {produce(a.query, a.max)} events")
    else:
        consume()
