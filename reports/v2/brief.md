# Banking77 — v1.2 brief

**56/56 quality conditions complete.** The extension adds equal training budgets for four supervised methods, retaining three zero-shot references and all original results.

| Method | Fit/class | K=5 | K=10 | K=15 | K=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + LR | 20 | 90.50% | 86.75% | 81.50% | 82.62% |
| TF-IDF + LR | 50 | 94.50% | 93.25% | 89.83% | 88.25% |
| TF-IDF + LR | 100 | 96.50% | 94.25% | 93.00% | 90.00% |
| MiniLM + LR | 20 | 93.50% | 93.25% | 92.83% | 88.50% |
| MiniLM + LR | 50 | 96.50% | 96.50% | 94.33% | 90.88% |
| MiniLM + LR | 100 | 98.00% | 96.50% | 95.83% | 92.38% |
| BERT | 20 | 56.00% | 21.25% | 17.67% | 16.25% |
| BERT | 50 | 72.50% | 58.75% | 51.33% | 67.88% |
| BERT | 100 | 96.00% | 92.50% | 87.67% | 85.12% |
| OpenAI zero-shot | 0 | 92.00% | 88.25% | 87.33% | 85.00% |
| Emissary routing zero-shot | 0 | 90.00% | 86.75% | 82.17% | 77.50% |
| Jev zero-shot | 0 | 94.00% | 92.25% | 89.83% | 86.12% |
| Emissary Qwen SFT | 20 | 84.00% | 91.50% | 89.50% | 84.50% |
| Emissary Qwen SFT | 100 | 98.00% | 93.25% | 93.83% | 92.25% |

At equal 20/class budgets, MiniLM+LR has the highest accuracy and macro-F1 among the four supervised methods at every K. At equal 100/class budgets, MiniLM leads accuracy at K=10/15/20 and ties Qwen at K=5 (98.00%); Qwen has the slightly higher macro-F1 at K=5. These are point estimates; paired intervals below quantify uncertainty.

Qwen 100/class no longer has the highest accuracy at K=20 after adding MiniLM 100/class: 92.25% versus 92.38%. The difference is one prediction out of 800. At K=10/15, MiniLM 100/class scores 96.50%/95.83%, versus Qwen 93.25%/93.83%.

Qwen 20/class scores 84.00%, 91.50%, 89.50% and 84.50% accuracy. More labels improve Qwen in every K under this fixed one-epoch recipe. The low-budget K=5 result shows that quality is not monotonic in class cardinality.

BERT at 20/class scores 56.00%, 21.25%, 17.67% and 16.25% accuracy, while its 100/class variants score 96.00%, 92.50%, 87.67% and 85.12%. This characterizes the fixed two-epoch recipe; it does not establish the best achievable performance of a separately tuned BERT model.

Offline checks passed for 16 historical supervised conditions and all 28 new conditions. There is no overlap of IDs, exact text, or text normalized with NFKC/casefold/collapsed whitespace between fit, reserved validation and test. Checks include the entire official test split and the recorded Qwen training uploads. [Audit evidence](data_integrity.json). This does not audit base-model pretraining, semantic duplicates or the provider’s internal processing.

OpenAI has no probabilities. Jev retains complete accuracy/F1 coverage but has 1/2/3 unavailable probability vectors at K=10/15/20; full-cohort calibration is unavailable there. The earlier recovery kept the 0.0001 sum tolerance and retained a class choice only when the sum was the sole invalidity and the choice was a maximum of finite nonnegative scores with positive total mass. No probabilities were renormalized. Recovery requested 970 missing IDs; historical and recovered timings remain separate.

One seed and previously observed test results make this an exploratory extension. Bootstrap intervals condition on fitted models, labels, class balance and campaign seed; they do not estimate training-seed variability. The 2,000 gold-stratified resamples use seed 20260925 and pointwise 95% percentile intervals without multiplicity adjustment. Test cohorts overlap across K and must not be pooled as independent observations. Definitions were assistant-reviewed, not independently human-validated. Changing K adds classes and examples, so trends do not isolate a causal cardinality effect.

Emissary pricing remains unknown. This brief makes no cross-provider cost or latency claim. See the [extended report](report.md) for full calibration results, paired uncertainty, measurement scope and reproduction.
