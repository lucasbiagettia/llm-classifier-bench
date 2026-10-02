# Banking77 — v1.2 with Llama supplement: brief

**64/64 quality conditions complete.** The extension includes equal training budgets for five supervised methods, retaining three zero-shot references and all original results.

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
| Emissary Llama SFT | 20 | 91.00% | 91.75% | 88.00% | 87.25% |
| Emissary Llama SFT | 100 | 98.00% | 95.25% | 95.00% | 93.38% |

At equal 20/class budgets, MiniLM+LR retains the highest accuracy and macro-F1 among all five supervised methods at every K. At 100/class, MiniLM leads accuracy at K=10/15; MiniLM, Qwen and Llama tie at K=5 (98.00%). Llama now leads at K=20 (93.375%, versus MiniLM 92.375% and Qwen 92.25%). These are point estimates; the paired intervals in the extended report quantify uncertainty.

Llama 100/class scores 98.00%, 95.25%, 95.00% and 93.375% accuracy. It ties Qwen at K=5 and exceeds it by 2.00, 1.17 and 1.13 percentage points at K=10/15/20. The eight equal-budget Llama-minus-Qwen accuracy intervals include zero except the 20/class K=5 and K=20 comparisons. The 100/class K=20 lead over MiniLM is also not resolved by its 95% paired interval.

Against Jev zero-shot, Llama 100/class improves accuracy by 4.00, 3.00, 5.17 and 7.25 percentage points; all four pointwise 95% paired accuracy intervals exclude zero. At K=20, that is 93.375% versus 86.125%, or 58 more correct predictions out of 800. This comparison uses different amounts of supervision and is not an equal-label-budget provider ranking.

Llama 20/class scores 91.00%, 91.75%, 88.00% and 87.25% accuracy. More labels improve Llama at every K. Its 20/class result exceeds Jev only at K=20, where the paired accuracy interval includes zero. Qwen 20/class remains at 84.00%, 91.50%, 89.50% and 84.50%.

Probability-quality leadership is now shared by Emissary SFT variants: Qwen 100/class has the lowest log loss at K=5/10 and Brier at K=5; Llama 100/class has the lowest log loss at K=15/20 and Brier at K=10/15/20 among variants with complete probabilities. ECE rankings vary; low ECE alone does not imply high accuracy. These are point rankings, without uncertainty tests for calibration or proper scoring rules. The regressions were not recalibrated.

BERT at 20/class scores 56.00%, 21.25%, 17.67% and 16.25% accuracy, while its 100/class variants score 96.00%, 92.50%, 87.67% and 85.12%. This characterizes the fixed two-epoch recipe; it does not establish the best achievable performance of a separately tuned BERT model.

Llama uses byte-identical uploads and test cohorts to the paired Qwen condition. The request omitted `max_grad_norm` and `warmup_ratio`, which the Llama parameter template did not expose. All eight successful training responses nevertheless report 0.3 and 0.03, respectively, and their complete reported hyperparameters match Qwen. This is provider-reported configuration, not an inspection of training internals. Llama is an instruction-tuned 1B base; Qwen is a 4B pretrained base, so this comparison does not isolate model size or instruction tuning. The earlier blocked preflight is retained separately and produced no training job or predictions.

Offline checks passed for 16 historical supervised conditions, 28 budget-extension conditions and eight Llama conditions (52 supervised conditions total). There is no overlap of IDs, exact text, or text normalized with NFKC/casefold/collapsed whitespace between fit, reserved validation and test. Checks include the entire official test split and the recorded Qwen and Llama training uploads. [Audit evidence](data_integrity.json). This does not audit base-model pretraining, semantic duplicates or the provider’s internal processing.

OpenAI has no probabilities. Jev retains complete accuracy/F1 coverage but has 1/2/3 unavailable probability vectors at K=10/15/20; full-cohort calibration is unavailable there. The earlier recovery kept the 0.0001 sum tolerance and retained a class choice only when the sum was the sole invalidity and the choice was a maximum of finite nonnegative scores with positive total mass. No probabilities were renormalized. Recovery requested 970 missing IDs; historical and recovered timings remain separate.

One seed and previously observed test results make this an exploratory extension. Bootstrap intervals condition on fitted models, labels, class balance and campaign seed; they do not estimate training-seed variability. The 2,000 gold-stratified resamples use seed 20260925 and pointwise 95% percentile intervals without multiplicity adjustment. Test cohorts overlap across K and must not be pooled as independent observations. Definitions were assistant-reviewed, not independently human-validated. Changing K adds classes and examples, so trends do not isolate a causal cardinality effect.

Emissary pricing remains unknown. This brief makes no cross-provider cost or latency claim. See the [extended report](report.md) for full calibration results, paired uncertainty, measurement scope and reproduction.
