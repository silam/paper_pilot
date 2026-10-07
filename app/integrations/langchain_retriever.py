"""
LangChain integration: expose our hybrid retriever as a LangChain `BaseRetriever`.

Why: LangGraph orchestrates our pipeline, but teams often have existing LangChain
chains (LCEL). Wrapping our retriever in the standard interface means ANY
LangChain component (chains, agents, evaluators, LangSmith tracing) can use it -
without giving up control of our own retrieval logic.

Example (LCEL):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_anthropic import ChatAnthropic
    retriever = PaperPilotRetriever(k=4)
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
             | ChatPromptTemplate.from_template("Context:{context}\\nQ:{question}")
             | ChatAnthropic(model="claude-sonnet-4-5"))
    chain.invoke("What is LoRA?")
"""
from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.retrieval import hybrid_search


class PaperPilotRetriever(BaseRetriever):
    k: int = 5
    filters: dict | None = None

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [Document(page_content=h.text, metadata={**h.metadata, "score": h.score})
                for h in hybrid_search(query, k=self.k, where=self.filters)]


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))
