"""
LlamaIndex alternative pipeline (for comparison / interview discussion).

LangChain/LangGraph = general orchestration (agents, graphs, tools).
LlamaIndex          = data-framework focus: loaders (LlamaHub), node parsers,
                      index types, query engines, response synthesizers.

This script builds the same corpus with LlamaIndex + the same Chroma server,
so you can run both and compare answers in the eval harness.

    pip install -r requirements-extras.txt
    python -m app.integrations.llamaindex_pipeline "What is PagedAttention?"
"""
from __future__ import annotations

import json
import sys

from app.ingestion.pipeline import SAMPLE


def build_query_engine():
    import chromadb
    from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=32)   # tokens, not chars

    papers = json.loads(SAMPLE.read_text())
    docs = [Document(text=p["text"], metadata={"title": p["title"], "doc_id": p["id"]}) for p in papers]

    client = chromadb.PersistentClient(path="./data/chroma_llamaindex")
    store = ChromaVectorStore(chroma_collection=client.get_or_create_collection("papers_li"))
    index = VectorStoreIndex.from_documents(
        docs, storage_context=StorageContext.from_defaults(vector_store=store))
    # Uses Settings.llm (OpenAI by default; set llama_index.llms.anthropic for Claude)
    return index.as_query_engine(similarity_top_k=4)


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What is PagedAttention?"
    response = build_query_engine().query(question)
    print(response)
    for n in response.source_nodes:
        print(f"- {n.metadata['title']} (score={n.score:.3f})")
