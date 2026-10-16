"""
LLM client - one interface, many providers (Model Serving abstraction).

Providers:
  * anthropic - Claude via the Anthropic SDK
  * openai    - OpenAI SDK
  * vllm      - self-hosted open model served by vLLM (or Hugging Face TGI);
                both expose an OpenAI-compatible /v1 API, so we reuse the OpenAI
                SDK with a different base_url. This is how you'd serve Llama /
                Mistral on your own GPUs to cut cost or keep data on-prem.
  * offline   - an extractive "LLM" (picks the most relevant sentences and cites
                them). No key, no cost -> the whole system runs in CI and demos.

Every call returns token usage + estimated cost, which feeds the Prometheus
cost counter and the per-request trace (Cost & Latency Optimization).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.observability import COST, TOKENS

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        print("*********** LLMResult.cost_usd")
        if self.model.startswith("offline"):
            return 0.0                      # local extractive mode is free
        s = get_settings()
        return (self.input_tokens * s.price_input_per_m + self.output_tokens * s.price_output_per_m) / 1e6


class LLMClient:
    def __init__(self):
        print("*********** LLMClient.__init__")
        self.s = get_settings()
        self.provider = self.s.llm_provider
        if self.provider == "anthropic":
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=self.s.anthropic_api_key)
        elif self.provider in ("openai", "vllm"):
            from openai import OpenAI
            if self.provider == "openai":
                self._openai = OpenAI(api_key=self.s.openai_api_key)
            else:
                self._openai = OpenAI(base_url=self.s.vllm_base_url, api_key="not-needed")

    @property
    def model_name(self) -> str:
        print("*********** LLMClient.model_name")
        return {"anthropic": self.s.anthropic_model, "openai": self.s.openai_model,
                "vllm": self.s.vllm_model}.get(self.provider, "offline-extractive")


    def complete(self, system: str, user: str, max_tokens: int | None = None,
                 retries: int = 2) -> LLMResult:
        """Call the model with retry + timing + token accounting."""
        print("*********** LLMClient.complete")
        t0 = time.perf_counter()
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                result = self._call(system, user, max_tokens or self.s.llm_max_tokens)

                print(f"############## LLM call succeeded on attempt {result}")

                break
            except Exception as exc:  # network / rate-limit errors
                last_err = exc
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"LLM unavailable: {last_err}")
        
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        TOKENS.labels("input").inc(result.input_tokens)
        TOKENS.labels("output").inc(result.output_tokens)
        COST.inc(result.cost_usd)
        return result

    def _call(self, system: str, user: str, max_tokens: int) -> LLMResult:
        print("*********** LLMClient._call")
        if self.provider == "anthropic":
            r = self._anthropic.messages.create(
                model=self.s.anthropic_model, system=system, max_tokens=max_tokens,
                temperature=self.s.llm_temperature,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return LLMResult(text, self.model_name, r.usage.input_tokens, r.usage.output_tokens)
        if self.provider in ("openai", "vllm"):
            r = self._openai.chat.completions.create(
                model=self.model_name, max_tokens=max_tokens, temperature=self.s.llm_temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            u = r.usage
            return LLMResult(r.choices[0].message.content or "", self.model_name,
                             u.prompt_tokens if u else 0, u.completion_tokens if u else 0)
        return offline_complete(system, user)


# ------------------------------------------------------------------ offline mode
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in and or for on with is are was were be by as at from that this it its "
            "what how why which who does do can use used using".split())


def _words(t: str) -> set[str]:
    print("*********** _words")
    return {w for w in _WORD.findall(t.lower()) if w not in _STOP and len(w) > 2}


def offline_complete(system: str, user: str) -> LLMResult:
    """Extractive answerer: picks the context sentences that best overlap the question."""
    print("*********** offline_complete")
    est_in = len(system + user) // 4
    if "<context>" not in user:                       # rewrite / judge prompts
        q = user.split("Question:", 1)[-1].split("Search query:")[0].strip()
        return LLMResult(q or user[:200], "offline-extractive", est_in, len(q) // 4)

    context = user.split("<context>", 1)[1].split("</context>", 1)[0]
    question = user.split("Question:", 1)[1].split("Answer", 1)[0]
    qwords = _words(question)
    scored = []
    for block in re.split(r"\n(?=\[\d+\])", context.strip()):
        m = re.match(r"\[(\d+)\]", block)
        if not m:
            continue
        n = m.group(1)
        body = block.split("\n", 2)[-1]
        for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
            if "[removed:" in sent:
                continue
            if sent.startswith("Title:") or len(sent) < 30:
                continue
            overlap = len(qwords & _words(sent))
            if overlap:
                scored.append((overlap / (len(qwords) or 1), sent.strip(), n))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] < 0.2:
        text = "I don't know based on the indexed papers."
    else:
        picked, seen = [], set()
        for _, sent, n in scored:
            if sent not in seen:
                picked.append(f"{sent} [{n}]")
                seen.add(sent)
            if len(picked) == 3:
                break
        text = " ".join(picked)
    return LLMResult(text, "offline-extractive", est_in, len(text) // 4)


@lru_cache
def get_llm() -> LLMClient:
    print("*********** get_llm")
    return LLMClient()
