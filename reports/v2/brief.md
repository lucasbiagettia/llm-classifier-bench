# Banking77 v2 — brief

**Evaluation results: 28/28 conditions complete.** We compare three zero-shot services and four supervised classifiers across 5, 10, 15 and 20 labels, using 40 held-out examples per label and one seed.

| Method | Accuracy K=5 | Accuracy K=20 | Change (pp) |
| --- | ---: | ---: | ---: |
| TF-IDF + LR (50/class) | 94.5% | 88.2% | -6.250 |
| MiniLM + LR (50/class) | 96.5% | 90.9% | -5.625 |
| BERT (50/class) | 72.5% | 67.9% | -4.625 |
| OpenAI zero-shot | 92.0% | 85.0% | -7.000 |
| Emissary routing zero-shot | 90.0% | 77.5% | -12.500 |
| Jev zero-shot | 94.0% | 86.1% | -7.875 |
| Emissary Qwen SFT (100/class) | 98.0% | 92.2% | -5.750 |

The table describes this selected label sequence; adding classes also changes the test population. It does not establish a universal or causal cardinality effect. Confidence intervals, macro-F1 and calibration metrics are in the [extended report](report.md).

**Supervision differs:** BERT, MiniLM+LR and TF-IDF+LR use 50 training examples/class. Emissary Qwen SFT uses 100/class, and also changes the model and mechanism relative to Emissary routing.

Jev’s initial K=10/15/20 runs were incomplete due to invalid probability sums. Recovery retains valid class choices with unavailable probabilities when only the sum is invalid; it never silently renormalizes. Missing conditions are not scored, and full-cohort calibration is unavailable when distributions are missing.

One seed, prior test exposure and historical output reuse limit generalization. The full report documents conditional bootstrap intervals, failures, phase-specific measurements and reproduction. Emissary costs remain unknown. This is not a v2.0.0 release announcement.
