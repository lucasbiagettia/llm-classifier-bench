# Banking77 v2 — extended report

**Evaluated: 28/28 quality conditions complete.** Missing conditions are not scored. This report is generated from saved evidence; it does not claim a v2.0.0 release.

## Design

One seed (42), nested label sets of 5/10/15/20 classes, and all 40 held-out test examples/class: 200/400/600/800 predictions per method. Class descriptions and source revision are frozen in [the protocol](protocol.md). The support pool contains 42 eligible labels. Training and test partitions remain separate.

OpenAI, Jev and Emissary routing are zero-shot. TF-IDF+LR and frozen MiniLM+LR use 50 fit examples/class with C=1; BERT uses 50/class and two CPU epochs. Emissary Qwen3-4B-Base SFT uses **100/class and one epoch**, so its comparisons with local supervised baselines do not hold supervision constant. Provider-internal use of the submitted rows is not independently audited. Quick Train remains unavailable and is not replaced by SFT.

The Qwen arm changes both model and mechanism relative to Emissary routing. The final model/settings inventory is in [the manifest](manifest.json). Checkpoint selection is the last checkpoint from the one-epoch job, without test-based selection.

## Quality and uncertainty

Accuracy intervals are 95% percentile bootstrap intervals, 2,000 resamples stratified by gold class, seed 20260925. Macro-F1 intervals and paired differences against OpenAI are saved in `results.json`. These condition on the fitted model, selected labels and one seed; they are pointwise, exploratory and not adjusted for multiple comparisons.

| Method | K | Coverage | Accuracy [95% CI] | Macro-F1 | ECE | Adaptive ECE | Log loss | Brier |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + LR (50/class) | 5 | 200/200 (completed) | 94.5% [91.5%, 97.5%] | 0.945 | 0.356 | 0.356 | 0.600 | 0.264 |
| TF-IDF + LR (50/class) | 10 | 400/400 (completed) | 93.2% [90.8%, 95.5%] | 0.932 | 0.500 | 0.500 | 0.957 | 0.409 |
| TF-IDF + LR (50/class) | 15 | 600/600 (completed) | 89.8% [87.3%, 92.2%] | 0.899 | 0.544 | 0.544 | 1.206 | 0.498 |
| TF-IDF + LR (50/class) | 20 | 800/800 (completed) | 88.2% [86.0%, 90.4%] | 0.884 | 0.577 | 0.577 | 1.391 | 0.559 |
| MiniLM + LR (50/class) | 5 | 200/200 (completed) | 96.5% [94.0%, 98.5%] | 0.965 | 0.169 | 0.162 | 0.248 | 0.086 |
| MiniLM + LR (50/class) | 10 | 400/400 (completed) | 96.5% [94.8%, 98.2%] | 0.966 | 0.290 | 0.289 | 0.452 | 0.175 |
| MiniLM + LR (50/class) | 15 | 600/600 (completed) | 94.3% [92.5%, 96.2%] | 0.943 | 0.318 | 0.318 | 0.547 | 0.213 |
| MiniLM + LR (50/class) | 20 | 800/800 (completed) | 90.9% [89.0%, 92.6%] | 0.909 | 0.330 | 0.330 | 0.678 | 0.271 |
| BERT (50/class) | 5 | 200/200 (completed) | 72.5% [68.5%, 76.0%] | 0.655 | 0.357 | 0.357 | 1.114 | 0.565 |
| BERT (50/class) | 10 | 400/400 (completed) | 58.8% [55.2%, 62.3%] | 0.565 | 0.357 | 0.357 | 1.630 | 0.738 |
| BERT (50/class) | 15 | 600/600 (completed) | 51.3% [48.3%, 54.2%] | 0.470 | 0.335 | 0.335 | 1.936 | 0.795 |
| BERT (50/class) | 20 | 800/800 (completed) | 67.9% [65.4%, 70.3%] | 0.656 | 0.484 | 0.484 | 1.818 | 0.735 |
| OpenAI zero-shot | 5 | 200/200 (completed) | 92.0% [88.5%, 95.5%] | 0.919 | — | — | — | — |
| OpenAI zero-shot | 10 | 400/400 (completed) | 88.2% [85.5%, 91.0%] | 0.880 | — | — | — | — |
| OpenAI zero-shot | 15 | 600/600 (completed) | 87.3% [84.8%, 89.7%] | 0.874 | — | — | — | — |
| OpenAI zero-shot | 20 | 800/800 (completed) | 85.0% [82.8%, 87.1%] | 0.847 | — | — | — | — |
| Emissary routing zero-shot | 5 | 200/200 (completed) | 90.0% [86.0%, 93.5%] | 0.897 | 0.056 | 0.034 | 0.366 | 0.170 |
| Emissary routing zero-shot | 10 | 400/400 (completed) | 86.8% [83.8%, 89.8%] | 0.866 | 0.046 | 0.047 | 0.535 | 0.197 |
| Emissary routing zero-shot | 15 | 600/600 (completed) | 82.2% [79.3%, 85.0%] | 0.826 | 0.068 | 0.060 | 0.707 | 0.258 |
| Emissary routing zero-shot | 20 | 800/800 (completed) | 77.5% [74.9%, 80.0%] | 0.775 | 0.029 | 0.033 | 0.951 | 0.343 |
| Jev zero-shot | 5 | 200/200 (completed) | 94.0% [90.5%, 97.0%] | 0.940 | 0.042 | 0.042 | 0.621 | 0.099 |
| Jev zero-shot | 10 | 400/400 (completed) | 92.2% [89.8%, 94.8%] | 0.923 | — | — | — | — |
| Jev zero-shot | 15 | 600/600 (completed) | 89.8% [87.5%, 92.2%] | 0.899 | — | — | — | — |
| Jev zero-shot | 20 | 800/800 (completed) | 86.1% [84.0%, 88.1%] | 0.860 | — | — | — | — |
| Emissary Qwen SFT (100/class) | 5 | 200/200 (completed) | 98.0% [96.0%, 99.5%] | 0.980 | 0.008 | 0.008 | 0.064 | 0.030 |
| Emissary Qwen SFT (100/class) | 10 | 400/400 (completed) | 93.2% [90.5%, 95.5%] | 0.933 | 0.026 | 0.024 | 0.208 | 0.099 |
| Emissary Qwen SFT (100/class) | 15 | 600/600 (completed) | 93.8% [92.0%, 95.7%] | 0.938 | 0.038 | 0.031 | 0.241 | 0.098 |
| Emissary Qwen SFT (100/class) | 20 | 800/800 (completed) | 92.2% [90.5%, 93.9%] | 0.922 | 0.027 | 0.025 | 0.288 | 0.115 |

## Cardinality observations

| Method | Accuracy K=5 | Accuracy K=20 | Change (percentage points) |
| --- | ---: | ---: | ---: |
| TF-IDF + LR (50/class) | 94.5% | 88.2% | -6.250 |
| MiniLM + LR (50/class) | 96.5% | 90.9% | -5.625 |
| BERT (50/class) | 72.5% | 67.9% | -4.625 |
| OpenAI zero-shot | 92.0% | 85.0% | -7.000 |
| Emissary routing zero-shot | 90.0% | 77.5% | -12.500 |
| Jev zero-shot | 94.0% | 86.1% | -7.875 |
| Emissary Qwen SFT (100/class) | 98.0% | 92.2% | -5.750 |

The K=5→20 changes are descriptive. Increasing K also adds new labels and test examples, so the difference does not isolate a causal effect of cardinality. Overlapping cohorts must not be pooled as independent observations. A general claim that zero-shot necessarily degrades is not established by this single-seed experiment.

![Accuracy by cardinality](accuracy.png)

## Failures, recovery and probability validity

The first small run completed 21/24 executable conditions. Jev stopped at 381/400, 3/600 and 446/800 valid predictions for K=10/15/20 because a returned probability vector failed the sum-to-one tolerance of 0.0001. Its old invalid vectors were not persisted, so their exact deviations cannot be reconstructed. Recovery requested only the 970 missing IDs.

The follow-up preserves the strict probability tolerance. If only the sum check fails but the provider choice is a valid maximum-score label and the nonnegative finite scores have positive total mass, the label is retained for accuracy/F1. Confidence and probabilities are marked unavailable; raw scores and sums are saved in `probability_diagnostics.jsonl`. No renormalization is performed. Other invalid responses still fail. Full-cohort calibration metrics are suppressed if any distribution is unavailable. This explicitly changes the acceptance policy for label metrics in the recovery phase.

In the completed recovery, Jev has 1/2/3 unavailable probability vectors at K=10/15/20. All labels are present, so accuracy and macro-F1 cover the full test sets; calibration is unavailable for those three conditions.

## Operational measurements and costs

All inference is singleton, concurrency 1, without warmup or automatic retries. Historical and fresh measurements are not pooled. `results.json` lists separate phase sample counts and p50/p95 client latency with source paths. Individual runs retain timings, usage (including failed calls), preparation and cost reports. Provider hardware and hosted cache state are uncontrolled. Unknown Emissary charges are unavailable, not zero. No total billed cost is asserted. Reused Jev evidence may span more than one historical attempt; its phase latency is descriptive only.

Original BERT fits all completed, totaling approximately 14 minutes; the 20-class condition took about 337 seconds. Qwen training and deployment timing are separate from inference and local BERT timings.

## Limitations and reproducibility

This revised campaign is exploratory: test results were previously observed, one seed cannot estimate between-seed variability, and definitions were reviewed by the assistant rather than independently validated by a human. Initial reuse retained complete historical runs; subsequent Jev recovery retains validated partial outputs. Disclose this availability selection when discussing reliability.

All 28 agreed quality conditions completed. The follow-up used four existing Qwen jobs and 970 additional Jev requests; original predictions were retained. Raw logs and deployment/job IDs remain in the evidence directories. No additional inference is needed to regenerate the reports.

Rebuild the tables, prose and figure from the versioned summary with `PYTHONPATH=src venv/bin/python scripts/build_v2_reports.py`. To recompute metrics and bootstrap intervals from original predictions, add `--recompute`; that requires the ignored `artifacts/` evidence bundle. The [manifest](manifest.json) records source/model revisions and label sets; `results.json` retains intervals and prediction fingerprints. Original per-run manifests retain exact IDs and provider resources. Local campaign launchers are not required to render the published results. No release tag is created by report generation.
