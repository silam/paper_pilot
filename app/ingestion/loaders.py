"""
EXTRACT step of the ETL pipeline: Data Collection & APIs + Document Loaders.

Sources supported:
  1. arXiv public API (Atom XML)  -> real data collection from an external API
  2. Local files: .pdf, .md, .txt, .json  -> classic "document loaders"

Every loader returns the same `RawDocument` shape, so downstream steps
(cleaning, chunking, embedding) don't care where data came from.
This is the "Adapter" pattern - add a new source by writing one function.

INTERVIEW TALKING POINT: "Loaders are pure functions that normalize heterogeneous
sources into one schema. Retries + rate-limit politeness for external APIs."
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass
class RawDocument:
    doc_id: str
    title: str
    text: str
    source: str                      # "arxiv" | "file"
    url: str = ""
    authors: list[str] = field(default_factory=list)
    published: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Used for idempotent ingestion: same content -> skip re-embedding."""
        return hashlib.sha256(self.text.encode()).hexdigest()


# ---------------------------------------------------------------- arXiv API
def fetch_arxiv(query: str, max_results: int = 20, retries: int = 3) -> list[RawDocument]:
    """Call the arXiv API with exponential backoff (APIs fail - plan for it)."""
    params = {"search_query": query, "start": 0, "max_results": max_results,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    for attempt in range(retries):
        try:
            resp = httpx.get(ARXIV_API, params=params, timeout=30)
            resp.raise_for_status()
            return parse_arxiv_atom(resp.text)
        except httpx.HTTPError as exc:
            wait = 2 ** attempt
            logger.warning("arxiv fetch failed (%s); retry in %ss", exc, wait)
            time.sleep(wait)
    raise RuntimeError("arXiv API unavailable after retries")


def parse_arxiv_atom(xml_text: str) -> list[RawDocument]:
    root = ET.fromstring(xml_text)
    docs = []
    for entry in root.findall(f"{ATOM}entry"):
        url = entry.findtext(f"{ATOM}id", "").strip()
        docs.append(RawDocument(
            doc_id="arxiv:" + url.rsplit("/", 1)[-1],
            title=" ".join(entry.findtext(f"{ATOM}title", "").split()),
            text=entry.findtext(f"{ATOM}summary", "").strip(),
            source="arxiv",
            url=url,
            authors=[a.findtext(f"{ATOM}name", "") for a in entry.findall(f"{ATOM}author")],
            published=entry.findtext(f"{ATOM}published", ""),
        ))
    return docs


# ---------------------------------------------------------------- local files
def load_file(path: Path) -> list[RawDocument]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        # A JSON file holding a list of papers (our bundled sample dataset).
        records = json.loads(path.read_text())
        return [RawDocument(
            doc_id=r["id"], title=r["title"], text=r["text"], source=r.get("source", "file"),
            url=r.get("url", ""), authors=r.get("authors", []), published=r.get("published", ""),
        ) for r in records]
    if suffix in {".md", ".txt"}:
        text = path.read_text(errors="ignore")
    elif suffix == ".pdf":
        from pypdf import PdfReader  # lazy import: optional dependency
        reader = PdfReader(str(path))
        # Keep page markers so citations can point to a page.
        text = "\n".join(f"[page {i + 1}]\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages))
    else:
        logger.info("skipping unsupported file %s", path)
        return []
    return [RawDocument(doc_id=f"file:{path.name}", title=path.stem, text=text, source="file")]


def load_directory(directory: str | Path) -> list[RawDocument]:
    docs: list[RawDocument] = []
    for p in sorted(Path(directory).rglob("*")):
        if p.is_file():
            docs.extend(load_file(p))
    return docs
