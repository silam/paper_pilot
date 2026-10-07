"""Test fixtures: isolated temp storage, offline LLM + hash embeddings (no keys, no network)."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="paperpilot-test-")
os.environ.update({
    "LLM_PROVIDER": "offline", "EMBEDDING_BACKEND": "hash",
    "CHROMA_PATH": f"{_TMP}/chroma", "DATABASE_URL": f"sqlite:///{_TMP}/test.db",
    "COLLECTION_NAME": "test_papers", "REDIS_URL": "",
})

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def indexed():
    """Ingest the bundled sample dataset once per test session."""
    from app.ingestion.loaders import load_file
    from app.ingestion.pipeline import SAMPLE, ingest_documents
    return ingest_documents(load_file(SAMPLE))


@pytest.fixture(scope="session")
def client(indexed):
    from fastapi.testclient import TestClient

    from app.main import app
    return TestClient(app)
