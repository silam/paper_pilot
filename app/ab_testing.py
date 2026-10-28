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

# Assign a user to a variant of an experiment based on deterministic bucketing
# user_id: unique identifier for the user
# experiment: name of the experiment to assign the user to
def assign_variant(user_id: str, experiment: str = "prompt_style") -> str:
    print("*********** assign_variant")

    # Ensure the experiment exists in the EXPERIMENTS dictionary
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")   
    variants = EXPERIMENTS[experiment]["variants"]

    # Compute a deterministic bucket for the user based on the experiment and user ID
    # Use SHA-256 to hash the combination of experiment and user ID, 
    # then map to a float in [0, 1)
    # Take the first 8 hex digits of the hash to get a 32-bit integer
    bucket = int(hashlib.sha256(f"{experiment}:{user_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0


    # Iterate over the variants and select the one corresponding to the user's bucket   
    for name, weight in variants.items():
        cumulative += weight
        if bucket <= cumulative:
            return name

    # Fallback: if no variant was selected (should not happen if weights sum to 1), return the first variant    
    # This should rarely happen, but ensures a variant is always returned
    return next(iter(variants))

# Perform a two-proportion z-test to compare the success rates of two variants
# success_a: number of successes for variant A
# n_a: total trials for variant A
# success_b: number of successes for variant B
# n_b: total trials for variant B
# Returns a dictionary with z-score, p-value, and significance at 0.05 level
def two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict:
    """Is variant B's success rate significantly different from A's?"""
    print("*********** two_proportion_z_test")
    if min(n_a, n_b) == 0:
        return {"z": 0.0, "p_value": 1.0}

    # Compute the success rates for each variant and the pooled success rate

    p_a, p_b = success_a / n_a, success_b / n_b

    # Compute the pooled success rate for the two variants
    p = (success_a + success_b) / (n_a + n_b)

    # Compute the standard error for the difference in success rates
    # Avoid division by zero by using a small epsilon if necessary
    se = math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b)) or 1e-9

    # Compute the z-score for the difference in success rates
    z = (p_b - p_a) / se

    # Compute the two-sided p-value for the z-score
    # Use the complementary error function to calculate the two-sided p-value
    # The two-sided p-value represents the probability of observing a z-score as extreme as the one computed, under the null hypothesis of no difference in success rates
    p_value = math.erfc(abs(z) / math.sqrt(2))     # two-sided


    # Determine if the result is statistically significant at the 0.05 level
    # A result is considered significant if the p-value is less than 0.05
    # Return the results including the success rates, z-score, p-value, and significance flag
    return {"rate_a": round(p_a, 3), "rate_b": round(p_b, 3), "z": round(z, 3),
            "p_value": round(p_value, 4), "significant_at_0.05": p_value < 0.05}
