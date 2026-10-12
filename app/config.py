"""
Central configuration (12-factor style).

WHY: Every setting comes from environment variables (or a .env file), so the
same Docker image runs locally, in CI, and in Kubernetes with different config.
pydantic-settings validates types at startup -> misconfiguration fails fast.

DISCUSSION TALKING POINT: "Config is code-free and validated. Secrets come from
env / K8s Secrets, never from the repo."
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- App ----
    app_name: str = "PaperPilot"
    environment: Literal["local", "ci", "staging", "prod"] = "local"
    log_level: str = "INFO"

    # ---- LLM ----
    # "anthropic" | "openai" | "vllm" (OpenAI-compatible server) | "offline"
    # "offline" = no API key needed; uses an extractive answerer so the whole
    # system (and CI) runs for free.
    llm_provider: Literal["anthropic", "openai", "vllm", "offline"] = "offline"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    vllm_base_url: str = "http://vllm:8000/v1"   # vLLM / TGI expose OpenAI-compatible APIs
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800

    # ---- Embeddings ----
    # "hf"   -> Hugging Face sentence-transformers (real semantic embeddings)
    # "hash" -> dependency-free hashing embedder (fast, deterministic; for tests/CI)
    embedding_backend: Literal["hf", "hash"] = "hash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ---- Vector store ----
    vector_backend: Literal["chroma", "pinecone"] = "chroma"
    chroma_mode: Literal["persistent", "http"] = "persistent"
    chroma_path: str = "./data/chroma"
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    collection_name: str = "papers"
    pinecone_api_key: str | None = None
    pinecone_index: str = "paperpilot"

    # ---- Chunking ----
    chunk_size: int = 800          # characters
    chunk_overlap: int = 120
    chunk_strategy: Literal["recursive", "sentence"] = "recursive"

    # ---- Retrieval ----
    top_k: int = 5
    hybrid_alpha: float = 0.5      # weight of vector vs BM25 in fusion
    min_relevance: float = 0.3     # below this, chunks are graded "irrelevant"

    # ---- Databases ----
    database_url: str = "sqlite:///./data/paperpilot.db"   # Postgres in docker
    redis_url: str | None = None                           # None -> in-memory cache
    cache_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.95

    # ---- Queues / streaming ----
    celery_broker_url: str = "redis://redis:6379/1"
    kafka_bootstrap: str = "kafka:9092"
    kafka_topic: str = "papers.raw"

    # ---- Safety ----
    enable_guardrails: bool = True
    hitl_confidence_threshold: float = 0.45   # below -> route to human review
    redact_pii: bool = True

    # ---- Cost tracking (USD per 1M tokens) ----
    price_input_per_m: float = 3.0
    price_output_per_m: float = 15.0


@lru_cache
def get_settings() -> Settings:
    """Cached singleton - settings are parsed once per process."""
    return Settings()
