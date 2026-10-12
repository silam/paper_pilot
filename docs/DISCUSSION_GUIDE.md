# DISCUSSION guide

## 1. Problem framing (open with this — 60 seconds)

> "Researchers and ML engineers lose hours finding *which* paper says *what*. General chatbots
> answer confidently but can't be trusted on specifics and don't cite. I built PaperPilot: a
> question-answering service over a curated paper corpus that **only answers from sources,
> cites every claim, says 'I don't know' otherwise, and escalates low-confidence answers to a
> human**. I treated it as a production system, not a notebook: ETL, evaluation gates in CI,
> observability, cost controls and safety."

**Success metrics** (say these — it's what product-minded engineers do):
- Quality: retrieval hit@5, groundedness, refusal accuracy on unanswerable questions
- User: thumbs-up rate, % answers needing human review, time-to-answer vs. manual search
- Cost/perf: p95 latency, $ per 1k queries, cache hit rate
- Safety: red-team block rate, false-block rate on benign questions

## 2. Five-minute live demo

| # | Show | Say |
|---|---|---|
| 1 | `python scripts/demo.py` sections 1–2 | "ETL is idempotent — content hashes mean re-runs cost nothing. PII is redacted before embedding." |
| 2 | Section 4 | "Hybrid retrieval: dense for meaning, BM25 for exact terms like 'PagedAttention', fused with RRF." |
| 3 | Section 5 | "Every answer is cited, scored for groundedness, and comes with a per-step trace." |
| 4 | Section 6 | "Repeat question → semantic cache → sub-millisecond, zero tokens." |
| 5 | Sections 7–9 | "Refuses unanswerable questions; blocks direct injection; neutralises a *poisoned document* — and its PII never got indexed." |
| 6 | Section 11 + UI review tab | "Thumbs-down → human review queue → corrected answers become eval cases; feedback also feeds the feature store." |
| 7 | `python -m eval.run_eval --gate` | "This runs in CI. A prompt change that drops groundedness can't merge." |
| 8 | Grafana screenshot / `k8s/` | "Same code runs in Compose and Kubernetes; HPA on the API, workers scale separately." |

Tip: before the DISCUSSION, run once with `LLM_PROVIDER=anthropic EMBEDDING_BACKEND=hf` and
take screenshots of the UI and Grafana for the README — real-model answers are more impressive
than the offline extractive mode.

## 3. Design decisions & trade-offs (be ready to defend)

| Decision | Why | Trade-off / alternative |
|---|---|---|
| LangGraph state machine instead of a linear chain | Retry and escalation are explicit, testable edges; each node is traceable | More boilerplate than a one-line chain |
| Hybrid + RRF | Dense misses exact identifiers; BM25 misses paraphrases | Two indexes to maintain; BM25 here is in-process (OK to ~1M chunks, then use OpenSearch/Weaviate hybrid) |
| Char-based recursive chunking, 800 / 120 overlap + title header | Simple, robust; header fixes "orphan chunks" | Semantic or structure-aware chunking may do better on long PDFs — decide with the eval set |
| Chroma behind an interface | Zero-ops locally, can swap to Pinecone/Weaviate | Chroma single-node limits at very large scale |
| Grade docs before generation | Irrelevant context is a top cause of hallucination; also saves tokens | Extra step; can wrongly drop a useful chunk (tune `min_relevance`) |
| Lexical groundedness check | Free, fast, catches fabricated entities | Misses paraphrased-but-wrong claims → use NLI model or LLM judge for high-stakes |
| Semantic cache at 0.95 | Big latency/cost win on FAQ-style traffic | Risk of reusing an answer for a subtly different question → high threshold, prompt version in key, flush on ingest |
| Offline mode | CI and demos need no keys; deterministic tests | Not representative of real answer quality |
| PII redaction at ingest | Erasure from embeddings is hard | Regex misses names/addresses → add Presidio/NER |

## 4. Likely questions — short answers

**How would you scale this to Meta-level traffic?**
Stateless API pods behind a load balancer with HPA; move BM25 to a distributed search engine;
shard the vector index (or managed vector DB) and use approximate search with tuned `ef_search`;
Redis cluster for cache; separate ingestion via Kafka with partitions = consumer parallelism;
self-host an open model on vLLM (continuous batching, PagedAttention) with a router sending easy
questions to a small model and hard ones to a large model; aggressive caching of embeddings and
answers; multi-region with data residency for GDPR.

**How do you know a change made things better?**
Offline: golden + red-team sets with layered metrics, compared per version (`--compare`), gated
in CI. Online: A/B by deterministic user bucketing, compare thumbs-up rate with a two-proportion
z-test, watch guardrail metrics (latency, cost, review rate) so a "better" prompt that doubles
cost doesn't ship.

**How do you detect hallucinations?**
Prevent (grounding prompt, refusal path, relevance grading) → detect (sentence-level support
score, citation validity, optional LLM judge/NLI) → contain (low confidence → human review, UI
shows confidence) → learn (reviewed failures become eval cases).

**Prompt injection?**
Two kinds: direct (user) and indirect (retrieved documents, web pages, tool outputs). Layers:
input classifier/patterns, delimiters + "context is data" instruction, sanitising retrieved
text, least-privilege tools (this system has no write tools exposed to the model), output
checks, monitoring for attack spikes. No single layer is sufficient.

**Where does latency go and how do you cut it?**
The trace shows it: generation dominates. Levers: cache, smaller/faster model for rewrite and
grading, streaming tokens to the UI, fewer/shorter chunks (also cheaper), prompt caching on
the provider side, parallelising retrieval calls, self-hosting with vLLM for throughput.

**How would you evaluate retrieval separately from generation?**
hit@k and MRR against the expected document per question (already implemented). If hit@k is
low, fix chunking/embeddings/hybrid weights before touching prompts.

**What are the weaknesses of this project?**
- Golden set is small (18 questions) — would grow it from production feedback and synthetic
  question generation, with human spot checks.
- Guardrails and groundedness are heuristic baselines.
- Offline mode's metrics are inflated (extractive answers are trivially grounded); real-model
  numbers are what matter.
- BM25 index is in-process per replica.
- No auth/multi-tenancy yet (would add OAuth/JWT and per-tenant metadata filters).

## 5. Ideas to extend (pick one before the DISCUSSION to show initiative)
1. Agentic mode: add a LangGraph tool node (arXiv search, citation graph lookup) with
   a ReAct loop and step limits.
2. Streaming responses with Server-Sent Events.
3. Fine-tune a small reranker or embedding model on collected feedback (LoRA).
4. Multimodal: extract figures/tables from PDFs.
5. Replace the regex input guard with a classifier and measure precision/recall on the
   red-team set.

## 6. Practice checklist
- [ ] Can draw the architecture from memory in under 2 minutes
- [ ] Can explain RRF, HNSW, cosine vs dot product, chunk overlap
- [ ] Can walk the `/ask` request through every node
- [ ] Can name 3 hallucination mitigations and 3 injection defenses
- [ ] Can state the metrics and why each matters to the business
- [ ] Have real-model screenshots + numbers in the README
