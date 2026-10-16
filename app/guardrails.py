"""
Guardrails: Prompt-Injection defense, PII protection, Hallucination detection.

Defense in depth - no single layer is enough:

  INPUT  guard  -> detect jailbreak / injection attempts in the user query
                   (pattern rules here; production adds a classifier such as
                   Llama Guard / Prompt Guard / a fine-tuned DeBERTa).
  CONTEXT guard -> retrieved documents are UNTRUSTED (indirect prompt injection:
                   a malicious paper could contain "ignore previous instructions").
                   We neutralise instruction-like lines and wrap context in tags.
  OUTPUT guard  -> (a) groundedness: is each answer sentence supported by the
                   sources?  (b) citation check: do cited [n] exist?
                   (c) PII leak check.

The groundedness score drives Human-in-the-Loop: low confidence -> queue for
human review instead of confidently returning a possibly hallucinated answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.ingestion.cleaning import redact_pii

# ------------------------------------------------------------------ input guard
INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now (?:in )?(dan|developer mode|jailbreak)",
    r"(reveal|print|show|leak) (me )?(your|the) (system prompt|instructions|api key)",
    r"pretend (you are|to be) .* without (any )?(rules|restrictions)",
    r"act as .* (no|without) (filters|guardrails)",
    r"</?(system|context)>",          # trying to forge our delimiters
]
_INJ = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class GuardResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    sanitized: str = ""


def check_input(query: str, max_len: int = 2000) -> GuardResult:
    print("*********** check_input")
    print(f"***********{query}")
    reasons = []
    if len(query) > max_len:
        reasons.append("query_too_long")
    for pat in _INJ:
        if pat.search(query):
            reasons.append("prompt_injection")
            break
    sanitized, pii = redact_pii(query)

    print(f"***********{sanitized}")
    if pii:
        reasons.append("pii_in_query_redacted")      # allowed, but redacted
    blocking = {"prompt_injection", "query_too_long"}
    return GuardResult(allowed=not (blocking & set(reasons)), reasons=reasons, sanitized=sanitized)


# ------------------------------------------------------------------ context guard
_INSTRUCTION_LINE = re.compile(
    r"^.*\b(ignore (all |the )?(previous|above)|system prompt|you must now|assistant:)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def sanitize_context(text: str) -> tuple[str, bool]:
    """Strip instruction-like lines from retrieved chunks (indirect injection)."""
    print("*********** sanitize_context")
    cleaned, n = _INSTRUCTION_LINE.subn("[removed: instruction-like text]", text)
    return cleaned, n > 0


# ------------------------------------------------------------------ output guard

_W = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in and or for on with is are was were be by as at from that this it its "
            "which can has have also into than these those their they what how why does who when where "
            "explain describe tell about paper papers".split())

"""
this function extracts the “important” words from a piece of text, 
ignoring common filler words and very short words. 
It returns a unique set of content words that can be used 
for filtering, scoring, similarity checks, or guardrail logic.

"""
def _content_words(t: str) -> set[str]:
    print("*********** _content_words")
    return {w for w in _W.findall(t.lower()) if w not in _STOP and len(w) > 2}

"""
estimates how well an answer is supported by the provided sources 
using word overlap; it does not call a model. 
If the answer contains “don't know” (case-insensitively),
 it treats that as a grounded refusal and immediately returns a score of 1.0 
 with no sentence details.

Otherwise, it converts each source into a set of content words, 
then splits the answer into sentences at whitespace following ., !, or ?. 
For each sentence longer than three characters, 
it removes citation markers such as [1], extracts its content words, 
and measures the fraction that appear in each source. The sentence’s support is the highest fraction from any one source, rounded to two decimal places. Sentences with no content words are skipped; sentences with support below 0.5 are marked unsupported.

The final score is the fraction of evaluated sentences 
that were not marked unsupported, 
rounded to three decimal places. 
If there are no evaluated sentences, 
the score is 0.0. 
Because this is based only on word overlap, 
it can miss paraphrases or 
flag claims that share words but contradict a source; 
it is a quick heuristic, not a semantic verification.
"""
def groundedness(answer: str, sources: list[str]) -> dict:
    """
    Fast, model-free hallucination check.
    For each answer sentence: support = max over sources of
        |content_words(sentence) ∩ content_words(source)| / |content_words(sentence)|
    A sentence is "supported" if support >= 0.5. Score = fraction supported.

    It's a cheap first line. For higher accuracy swap in an NLI model
    (e.g. a DeBERTa MNLI cross-encoder) or LLM-as-judge (see judge_groundedness).
    """
    print("*********** groundedness")
    if "don't know" in answer.lower():
        return {"score": 1.0, "sentences": [], "unsupported": []}   # refusal is grounded

    # It returns a unique set of content words that can be used 
    # for filtering, scoring, similarity checks, or guardrail logic
    src_words = [_content_words(s) for s in sources]
    details, unsupported = [], []

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(s) > 3]
    for sent in sentences:
        cw = _content_words(re.sub(r"\[\d+\]", "", sent))
        if not cw:
            continue
        
        support = max((len(cw & sw) / len(cw) for sw in src_words), default=0.0)
        
        details.append({"sentence": sent, "support": round(support, 2)})
        
        if support < 0.5:
            unsupported.append(sent)

    score = 1.0 - len(unsupported) / len(details) if details else 0.0
    
    return {"score": round(score, 3), "sentences": details, "unsupported": unsupported}

#------------------------------------------------------------------ citation guard
def check_citations(answer: str, n_sources: int) -> dict:
    print("*********** check_citations")
    cited = {int(c) for c in re.findall(r"\[(\d+)\]", answer)}
    invalid = sorted(c for c in cited if c < 1 or c > n_sources)
    return {"cited": sorted(cited), "invalid": invalid,
            "has_citations": bool(cited) or "don't know" in answer.lower()}


def check_output(answer: str) -> tuple[str, list[str]]:
    print("*********** check_output")
    redacted, pii = redact_pii(answer)
    return redacted, ([f"pii_leak:{k}" for k in pii] if pii else [])


def judge_groundedness(llm, answer: str, context: str) -> float | None:
    """LLM-as-judge variant (used when a real LLM is configured)."""
    print("*********** judge_groundedness")
    from app.llm.prompts import GROUNDEDNESS_JUDGE
    system, user = GROUNDEDNESS_JUDGE.render(context=context, answer=answer)
    try:
        raw = llm.complete(system, user, max_tokens=300).text
        return float(json.loads(raw[raw.find("{"): raw.rfind("}") + 1])["score"])
    except Exception:
        return None
