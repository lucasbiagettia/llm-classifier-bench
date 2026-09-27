# Banking77 v2: completed experimental design

The [extended report](report.md) and [brief](brief.md) cover **28 completed
conditions**: seven methods across 5, 10, 15 and 20 labels. The portable
[manifest](manifest.json) records the final configuration and source revisions.
Earlier pilot matrices and machine-specific launchers are local working material.

## Data and supervision

Banking77 source revision `57ec275d8078af65b7731c2a98be812d844a6d6b`, seed 42,
nested label sets, and all 40 held-out test examples per class. Each method has
200/400/600/800 predictions at K=5/10/15/20. Within each K the methods share the
same test IDs, text and gold labels. Across K, cohorts overlap and are not independent.

The eligible pool has 42 labels with at least 125 training examples/class. Reserve
25/class from the 125 sampled candidates, then select the method's fixed training
budget from the remaining 100. Reserved validation labels are not consumed.

| Method | Training labels/class | Fixed configuration |
| --- | ---: | --- |
| OpenAI | 0 | GPT-5 nano snapshot, minimal reasoning, no demonstrations |
| Emissary routing | 0 | Provider-managed routing with pinned experiment versions |
| Jev | 0 | jev-1.13.0 |
| TF-IDF + logistic regression | 50 | C=1, no validation search |
| Frozen MiniLM + logistic regression | 50 | all-MiniLM-L6-v2, C=1 |
| BERT | 50 | bert-base-uncased, two epochs, batch 16, LR 2e-5, max length 128 |
| Emissary Qwen SFT | 100 | Qwen3-4B-Base, one epoch, last checkpoint |

**100/class means 100 training shots/class.** SFT labels are not in-context
prompt demonstrations. Zero-shot and supervised conditions do not have equal
supervision; Qwen also has twice the labeled examples of the local baselines.
Changing from Emissary routing to Qwen SFT changes mechanism and base model.
Quick Train was unavailable through the recorded API and is excluded, not
substituted with SFT. Earlier Llama readiness probes are not reported as Qwen runs.

## Measurement and recovery

Singleton inference, concurrency 1, no warmup and no automatic retries. Model
loading, training and deployment are recorded separately from prediction latency.
Local methods used CPU with four numeric-library threads. The installed CUDA
build could not execute on the GTX 1050, so no GPU timings are claimed.

Compatible historical zero-shot outputs supplied 1700 initial predictions.
The first small matrix completed 21/24 conditions; Jev K=10/15/20 stopped on
invalid probability sums. The continuation retained 830 validated outputs and
requested 970 missing predictions. Four successful Qwen jobs supplied 2000 new
test predictions. Costs/timings remain in their original attempt ledgers;
historical and new phases are not silently pooled.

Jev initially required probabilities to sum to 1 within 0.0001. The follow-up
kept that tolerance, but retained the class label if only the sum check failed
and finite nonnegative scores had complete label coverage, positive total mass
and a valid maximum-score choice. Confidence/probabilities were then unavailable,
with original scores recorded for diagnosis. No renormalization was performed.
The final Jev K=10/15/20 cohorts have 1/2/3 such outputs; all labels are present,
but full-cohort calibration is unavailable. This policy change is disclosed
rather than retroactively described as part of the initial frozen design.

## Metrics and uncertainty

Accuracy, macro-F1, top-label ECE, adaptive ECE, log loss and Brier are reported
where supported. OpenAI provides no probabilities. Calibration is not estimated
from only the successful probability subset of an incomplete cohort.

Quality intervals use 2000 gold-stratified bootstrap resamples, seed 20260925,
95% percentile intervals. Paired differences against OpenAI share the same
resampled IDs within each K. These are pointwise exploratory intervals, conditional
on fixed labels, one fitted model and one seed, without multiplicity correction.
They do not estimate between-seed uncertainty or isolate a causal cardinality effect.

The one-seed design was an explicit scope reduction. Test results were seen
previously; definitions were assistant-reviewed, not independently human-validated.
The study is exploratory. Emissary pricing is unknown; no total billed spending
or fulfilled monetary ceiling is asserted.

## Reproduction

From a clean clone with dependencies installed, reproduce the published tables,
Markdown and figure using the committed aggregate summary, without APIs:

```bash
PYTHONPATH=src venv/bin/python scripts/build_v2_reports.py
```

For a full metric/interval recomputation, restore the original `artifacts/`
evidence directories listed in `manifest.json`, then run:

```bash
PYTHONPATH=src venv/bin/python scripts/build_v2_reports.py --recompute
```

The raw bundle is retained locally and is not distributed in Git. The manifest
contains immutable source CSV URLs and hashes, model revisions and selected label
sets; original configurations, prediction IDs, diagnostics, API usage and raw logs
remain in the bundle. `results.json` includes prediction hashes and all bootstrap
results. Reproducing hosted predictions requires provider access and can yield
different outputs; re-rendering saved results does not.

For a new independently budgeted campaign, use the maintained
`scripts/run_banking77_scaling_benchmark_v2.py --help` and the general experimental
protocol. Retired one-off launchers live in ignored `scripts/local/`; they are not
the public reproduction interface. This publication does not create v2.0.0.
