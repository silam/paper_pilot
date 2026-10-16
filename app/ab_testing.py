"""
Prompt A/B testing.

Deterministic bucketing: hash(user_id + experiment) -> [0,1). The same user
always sees the same variant (consistent UX, clean statistics), and no state
needs to be stored to know a user's assignment.

Results are analysed from SQL (db.prompt_version_report) and offline with
eval/run_eval.py --compare. Decide with a significance test, not vibes:
for thumbs-up rates use a two-proportion z-test (implemented below).
"""
from __future__ import annotations

import hashlib
import math

EXPERIMENTS = {
    "prompt_style": {"variants": {"rag_v1": 0.5, "rag_v2_fewshot": 0.5}},
}


def assign_variant(user_id: str, experiment: str = "prompt_style") -> str:
    print("*********** assign_variant")
    variants = EXPERIMENTS[experiment]["variants"]
    bucket = int(hashlib.sha256(f"{experiment}:{user_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, weight in variants.items():
        cumulative += weight
        if bucket <= cumulative:
            return name
    return next(iter(variants))


def two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict:
    """Is variant B's success rate significantly different from A's?"""
    print("*********** two_proportion_z_test")
    if min(n_a, n_b) == 0:
        return {"z": 0.0, "p_value": 1.0}
    p_a, p_b = success_a / n_a, success_b / n_b
    p = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b)) or 1e-9
    z = (p_b - p_a) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))     # two-sided
    return {"rate_a": round(p_a, 3), "rate_b": round(p_b, 3), "z": round(z, 3),
            "p_value": round(p_value, 4), "significant_at_0.05": p_value < 0.05}
