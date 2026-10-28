# Two-Proportion Z-Test in AI Engineering

A **two-proportion Z-test** is a statistical test used to determine whether two groups have significantly different success rates.

In AI engineering, it can help compare prompts, LLMs, RAG configurations, AI agents, guardrails, or production experiments.

## 1. Example: Prompt A/B Testing

Suppose you are building a chatbot that recommends Red Wing Shoes products and want to compare two prompts.

| Metric | Prompt A (existing) | Prompt B (improved) |
|---|---:|---:|
| Questions tested | 800 | 800 |
| Successful answers | 560 | 608 |
| Success rate | 70% | 76% |

Prompt B has a higher observed success rate. The statistical question is whether the difference is likely to reflect a real difference in performance rather than random variation.

## 2. Hypotheses

- **Null hypothesis (`H₀`)**: The two true success rates are equal.
- **Alternative hypothesis (`H₁`)**: The two true success rates are different.

The test calculates a **Z-score** and a **p-value**.

## 3. Formula

\[
Z = \frac{\hat p_1-\hat p_2}
{\sqrt{\hat p(1-\hat p)(\frac{1}{n_1}+\frac{1}{n_2})}}
\]

Where:

- \(\hat p_1\), \(\hat p_2\): observed success rates for groups 1 and 2.
- \(n_1\), \(n_2\): sample sizes.
- \(\hat p\): pooled success rate under the null hypothesis.

In practice, Python can calculate the test for you.

## 4. Python Example

### Install the dependency

```bash
pip install statsmodels
```

### Run the test

```python
from statsmodels.stats.proportion import proportions_ztest

# Prompt A: 560 successful answers out of 800 questions
success_a = 560
total_a = 800

# Prompt B: 608 successful answers out of 800 questions
success_b = 608
total_b = 800

counts = [success_a, success_b]
sample_sizes = [total_a, total_b]

z_score, p_value = proportions_ztest(
    counts,
    sample_sizes,
    alternative="two-sided",
)

print(f"Prompt A success rate: {success_a / total_a:.1%}")
print(f"Prompt B success rate: {success_b / total_b:.1%}")
print(f"Z-score: {z_score:.3f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05

if p_value < alpha:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")
```

### Expected interpretation

- Prompt A success rate: **70%**
- Prompt B success rate: **76%**
- The p-value is approximately **0.009**
- Because the p-value is below 0.05, the difference is statistically significant at the 5% significance level.

This provides evidence that the success rates differ, and Prompt B has the higher observed rate.

## 5. How to Interpret the P-Value

| Result | Interpretation |
|---|---|
| `p < 0.05` | Evidence against equal success rates, assuming the test conditions hold. |
| `p >= 0.05` | Insufficient evidence of a difference; this does **not** prove the systems are equal. |

A p-value of 0.009 does **not** mean there is a 99.1% probability that Prompt B is better. It means that, if the true success rates were equal and the assumptions held, a result at least this extreme would occur about 0.9% of the time.

## 6. AI Engineering Use Cases

| Use case | What you compare |
|---|---|
| Prompt A/B testing | Percentage of questions answered successfully by two prompts |
| LLM evaluation | Percentage of correct answers from two models |
| RAG evaluation | Percentage of questions answered correctly by two retrieval configurations |
| AI agents | Percentage of tasks completed successfully by two agent versions |
| Guardrail evaluation | Percentage of unsafe requests correctly blocked |
| Production experimentation | User conversion or task-completion rates before and after an AI feature change |

For example, you can compare two RAG chunking strategies by evaluating whether each produces a correct answer on a set of questions. Define the success criteria before running the experiment, then compare the proportions.

## 7. Important Assumptions and Caveats

- **Independent observations:** The basic two-proportion Z-test assumes observations are independent.
- **Adequate sample size:** The normal approximation needs sufficiently large expected success and failure counts in both groups.
- **Paired questions:** If the exact same questions are tested against both systems, the results may be paired. A paired test such as **McNemar's test** may be more appropriate.
- **Practical significance:** Statistical significance does not automatically mean an improvement is large enough to matter in production.
- **Evaluation quality:** Use representative test data and a consistent, preferably clearly defined success criterion.
- **Repeated experiments:** If you test many prompts or metrics, account for multiple comparisons and avoid selecting a winner based only on noisy results.

## 8. AI Engineer Interview Answer

**Question:** You changed an LLM prompt. The old prompt answers 65% of questions correctly, while the new prompt answers 72% correctly. How would you determine whether the new prompt is significantly better?

**Sample answer:**

> I would run an evaluation on a representative dataset and define what counts as a correct answer before testing. If the observations are independent and the sample-size assumptions are satisfied, I would compare the two success proportions using a two-proportion Z-test at a significance level such as 0.05. I would examine the p-value and confidence interval, then consider the size of the improvement, evaluation costs, and production impact before deciding whether to deploy the new prompt. If both prompts were evaluated on the same questions, I would consider a paired test such as McNemar's test.

## Key Takeaway

A **two-proportion Z-test** helps an AI engineer assess whether a difference between two observed success rates is supported by statistical evidence. It is useful for controlled comparisons, but the experimental design and quality of the evaluation data matter just as much as the test itself.
