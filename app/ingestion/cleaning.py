"""
TRANSFORM step (part 1): Data Cleaning & Preprocessing + Privacy.

Garbage in = garbage retrieved. Cleaning has an outsized effect on RAG quality:
  - fix hyphenated line breaks from PDFs ("trans-\\nformer" -> "transformer")
  - collapse whitespace, drop control characters
  - strip LaTeX noise that hurts embeddings
  - drop near-empty docs and duplicates
  - redact PII BEFORE it is embedded (GDPR/HIPAA: you can't easily "un-embed")

INTERVIEW TALKING POINT: "PII is redacted at ingestion time, not just at output
time, because once personal data lands in a vector index, deleting it for a
GDPR 'right to be forgotten' request is painful."
"""
from __future__ import annotations

import re
import unicodedata

from app.ingestion.loaders import RawDocument

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_LATEX_CMD = re.compile(r"\\(?:cite|ref|label|footnote)\{[^}]*\}")
_INLINE_MATH = re.compile(r"\$[^$]{1,80}\$")
_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")

# --- PII patterns (regex baseline; production would add an NER model like Presidio)
PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace PII with typed placeholders. Returns (clean_text, counts)."""
    counts: dict[str, int] = {}
    for label, pattern in PII_PATTERNS.items():
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return text, counts


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _LATEX_CMD.sub("", text)
    text = _INLINE_MATH.sub(" [math] ", text)
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def clean_documents(docs: list[RawDocument], redact: bool = True,
                    min_chars: int = 50) -> tuple[list[RawDocument], dict]:
    """Clean + dedupe + redact. Returns docs and a small data-quality report."""
    seen_hashes: set[str] = set()
    out: list[RawDocument] = []
    report = {"input": len(docs), "too_short": 0, "duplicates": 0, "pii_redacted": 0}
    for d in docs:
        d.text = clean_text(d.text)
        if redact:
            d.text, counts = redact_pii(d.text)
            report["pii_redacted"] += sum(counts.values())
        if len(d.text) < min_chars:
            report["too_short"] += 1
            continue
        if d.content_hash in seen_hashes:
            report["duplicates"] += 1
            continue
        seen_hashes.add(d.content_hash)
        out.append(d)
    report["output"] = len(out)
    return out, report
