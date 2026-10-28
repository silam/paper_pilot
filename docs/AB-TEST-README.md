# A/B Test Variant Assignment

Assigns users to experiment variants with **deterministic, hash-based bucketing**. The same user always gets the same variant for a given experiment, and you don't need a database, a session or any stored state to make that happen.

The default experiment, `prompt_style`, compares two RAG prompt strategies:

| Variant          | Weight | Description                          |
|------------------|--------|--------------------------------------|
| `rag_v1`         | 0.5    | Baseline RAG prompt                  |
| `rag_v2_fewshot` | 0.5    | RAG prompt with few-shot examples    |

---

## How it works

1. **Hash.** The function concatenates the experiment name and user ID (`"prompt_style:user-123"`) and hashes the result with SHA-256.
2. **Bucket.** It takes the first 8 hex digits (32 bits) of the hash and divides by `0xFFFFFFFF`, which gives a float `bucket` in `[0, 1]`.
3. **Select.** It walks the variants in order and keeps a running total of their weights. The first variant whose cumulative weight is `>= bucket` wins.

```
0.0                    0.5                    1.0
|------- rag_v1 -------|--- rag_v2_fewshot ---|
            ^ bucket = 0.31  →  rag_v1
```

### Why the experiment name is part of the hash

Because the experiment name is included in the hash, each experiment buckets users independently. A user in `rag_v1` for `prompt_style` has no built-in correlation with their variant in any other experiment, so experiments don't contaminate each other.

### Properties

- **Deterministic.** The same `(user_id, experiment)` pair always returns the same variant, across processes, servers and restarts.
- **Stateless.** No assignment table is needed.
- **Uniform.** SHA-256 output is evenly distributed, so traffic split closely matches the configured weights.
- **Order-sensitive.** Variant order in the dict defines bucket ranges. Reordering variants reassigns users (see [Caveats](#caveats)).

---

## Installation

The module only needs the standard library. Make sure `hashlib` is imported at the top of the file:

```python
import hashlib
```

> ⚠️ The snippet as written calls `hashlib.sha256` but doesn't import `hashlib`. Without the import it raises `NameError: name 'hashlib' is not defined` on the first call.

---

## API

### `assign_variant(user_id: str, experiment: str = "prompt_style") -> str`

| Parameter    | Type  | Description                                                 |
|--------------|-------|-------------------------------------------------------------|
| `user_id`    | `str` | Stable, unique identifier for the user                      |
| `experiment` | `str` | Key in `EXPERIMENTS`. Defaults to `"prompt_style"`          |

**Returns:** the name of the assigned variant (`str`).

**Raises:** `ValueError` if `experiment` isn't defined in `EXPERIMENTS`.

---

## Examples

### 1. Basic assignment

```python
from ab_test import assign_variant

assign_variant("user-123")           # 'rag_v2_fewshot'
assign_variant("user-456")           # 'rag_v1'
assign_variant("alice@example.com")  # 'rag_v2_fewshot'
assign_variant("42")                 # 'rag_v1'
```

### 2. Assignment is deterministic

```python
v1 = assign_variant("user-123")
v2 = assign_variant("user-123")
assert v1 == v2  # always True, even across restarts or different machines
```

### 3. Routing a RAG request by variant

```python
from ab_test import assign_variant

PROMPTS = {
    "rag_v1": (
        "Answer the question using the context below.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    ),
    "rag_v2_fewshot": (
        "Answer the question using the context below.\n\n"
        "Example:\nContext: The store opens at 9am.\n"
        "Question: When does the store open?\nAnswer: 9am.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    ),
}

def build_prompt(user_id: str, question: str, context: str) -> tuple[str, str]:
    variant = assign_variant(user_id, "prompt_style")
    prompt = PROMPTS[variant].format(context=context, question=question)
    return variant, prompt

variant, prompt = build_prompt("user-123", "What sizes are available?", "...")
# variant == 'rag_v2_fewshot'
```

### 4. Using it in a FastAPI endpoint

```python
from fastapi import FastAPI, Header

from ab_test import assign_variant

app = FastAPI()

@app.post("/chat")
def chat(question: str, x_user_id: str = Header(...)):
    variant = assign_variant(x_user_id)
    # ... run the RAG pipeline with the chosen prompt ...
    return {"variant": variant, "answer": "..."}
```

Return or log the variant with every response so outcomes can be attributed to it later.

### 5. Logging exposures for analysis

```python
import json
import logging
import time

logger = logging.getLogger("experiments")

def log_exposure(user_id: str, experiment: str, variant: str) -> None:
    logger.info(json.dumps({
        "event": "exposure",
        "ts": time.time(),
        "user_id": user_id,
        "experiment": experiment,
        "variant": variant,
    }))

variant = assign_variant("user-456")
log_exposure("user-456", "prompt_style", variant)
```

Join these exposure events with your outcome metrics (thumbs up/down, answer relevance score, latency, token cost) to compare variants.

### 6. Adding an experiment with an uneven split

```python
EXPERIMENTS["model_choice"] = {
    "variants": {"sonnet": 0.8, "haiku": 0.2},
}

assign_variant("user-123", "model_choice")
```

Weights should sum to `1.0`.

### 7. Three-way split

```python
EXPERIMENTS["chunk_size"] = {
    "variants": {"chunk_256": 0.34, "chunk_512": 0.33, "chunk_1024": 0.33},
}
```

### 8. Handling an unknown experiment

```python
try:
    assign_variant("user-123", "does_not_exist")
except ValueError as e:
    print(e)  # Unknown experiment: does_not_exist
```

### 9. Checking the traffic split

```python
from collections import Counter

from ab_test import EXPERIMENTS, assign_variant

print(Counter(assign_variant(f"user-{i}") for i in range(100_000)))
# Counter({'rag_v1': 50043, 'rag_v2_fewshot': 49957})

EXPERIMENTS["model_choice"] = {"variants": {"sonnet": 0.8, "haiku": 0.2}}
print(Counter(assign_variant(f"user-{i}", "model_choice") for i in range(100_000)))
# Counter({'sonnet': 80049, 'haiku': 19951})
```

The observed split stays within about 0.1% of the configured weights.

### 10. Unit tests (pytest)

```python
import pytest
from collections import Counter

from ab_test import EXPERIMENTS, assign_variant

def test_deterministic():
    assert assign_variant("user-1") == assign_variant("user-1")

def test_returns_known_variant():
    variants = EXPERIMENTS["prompt_style"]["variants"]
    for i in range(1_000):
        assert assign_variant(f"u{i}") in variants

def test_unknown_experiment_raises():
    with pytest.raises(ValueError, match="Unknown experiment"):
        assign_variant("user-1", "nope")

def test_split_is_close_to_weights():
    counts = Counter(assign_variant(f"u{i}") for i in range(20_000))
    share = counts["rag_v1"] / 20_000
    assert 0.48 < share < 0.52
```

---

## Caveats

- **Use a stable `user_id`.** If the ID changes (for example, an anonymous session ID that becomes a logged-in ID), the user may switch variants mid-experiment.
- **Don't reorder or reweight a running experiment.** Bucket ranges come from dict order and weights, so changes reassign existing users. To change the split, start a new experiment name (e.g. `prompt_style_v2`).
- **Weights should sum to 1.0.** If they sum to less, users with buckets above the total fall through to the first variant, which skews traffic toward it. If they sum to more, the last variant(s) get less traffic than configured. Consider validating at startup:

  ```python
  import math

  for name, cfg in EXPERIMENTS.items():
      total = sum(cfg["variants"].values())
      assert math.isclose(total, 1.0), f"{name} weights sum to {total}"
  ```

- **Remove the debug `print`.** `print("*********** assign_variant")` runs on every call. In production, drop it or replace it with `logger.debug(...)`.
- **Assignment isn't exposure.** Only log a user as "in the experiment" when they actually hit the code path that uses the variant. Otherwise your analysis gets diluted.
