# Banking77 v2: completed experimental design

The [extended report](report.md) and [brief](brief.md) cover **64 completed
conditions**: sixteen method/budget variants across 5, 10, 15 and 20 labels. The portable
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
| TF-IDF + logistic regression | 20/50/100 | C=1, no validation search |
| Frozen MiniLM + logistic regression | 20/50/100 | all-MiniLM-L6-v2, C=1 |
| BERT | 20/50/100 | bert-base-uncased, two epochs, batch 16, LR 2e-5, max length 128 |
| Emissary Qwen SFT | 20/100 | Qwen3-4B-Base, one epoch, last checkpoint |
| Emissary Llama SFT | 20/100 | Llama-3.2-1B-Instruct, one epoch, last checkpoint |

**100/class means 100 training shots/class.** SFT labels are not in-context
prompt demonstrations. Zero-shot and supervised conditions do not have equal
supervision. The new 20/100 variants now permit equal-budget supervised comparisons.
Within each K, pools are nested (20 ⊂ 50 ⊂ 100) and equal across methods. The
20-shot Qwen jobs start from the pretrained base, not the 100-shot checkpoints.
Epoch counts are fixed; more labels also mean more optimization steps.
Changing from Emissary routing to Qwen SFT changes mechanism and base model.
Quick Train was unavailable through the recorded API and is excluded, not
substituted with SFT. Earlier Llama readiness probes are not reported as Qwen runs.

## Offline data audit

All 16 historical supervised conditions, 28 budget-extension conditions and eight Llama conditions passed
ID, exact-text and normalized-text checks against the entire official test split.
Normalization uses NFKC, casefold and collapsed whitespace. Exact training pools,
consumed validation counts (zero), model/settings identity and the new Qwen uploads
and byte-identical Llama uploads were verified from saved artifacts. The [audit](data_integrity.json) is portable.
This does not audit base pretraining, semantic duplicates or provider internals.

## Measurement and recovery

Singleton inference, concurrency 1, no warmup and no automatic retries. Model
loading, training and deployment are recorded separately from prediction latency.
Local methods used CPU with four numeric-library threads. The installed CUDA
build could not execute on the GTX 1050, so no GPU timings are claimed.

Compatible historical zero-shot outputs supplied 1700 initial predictions.
The first small matrix completed 21/24 conditions; Jev K=10/15/20 stopped on
invalid probability sums. The continuation retained 830 validated outputs and
requested 970 missing predictions. Four successful Qwen jobs supplied 2000 new
test predictions. The budget extension completed 28/28 additional conditions,
14,000 predictions, four fresh Qwen training jobs and four deployments. Costs/timings remain in their original attempt ledgers;
historical and new phases are not silently pooled. The Llama supplement adds eight successful
fresh jobs/deployments and 4,000 predictions; the complete report has 64 conditions
and 32,000 predictions. Its first local preflight stopped before any upload or
training and is retained separately. The earlier September readiness probes remain
separate from these eight successful jobs.

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
resampled IDs within each K. The update also includes matched-budget contrasts
against MiniLM, within-method 100-minus-20 effects, Llama-minus-Qwen at equal
budgets, and Emissary-minus-Jev contrasts (unequal supervision for SFT). These are pointwise exploratory intervals, conditional
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
evidence directories listed in `manifest.json`, including the budget extension, then run:

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
the public reproduction interface. Publication version `v1.2` names this report
update; the experiment continues to use the `reports/v2` directory.

## Llama supplement configuration

The Llama requests use the same fit pools, upload bytes and shared parameters
as Qwen. `max_grad_norm` and `warmup_ratio` were omitted because the Llama
parameter template did not expose them. Every successful training response
reports 0.3 and 0.03, respectively, and the full reported hyperparameters match
Qwen in all eight conditions. This checks saved provider responses, not internal
training implementation. Llama is instruction-tuned and Qwen is a pretrained
base; this does not isolate model size or instruction tuning.

The immutable `v1.2` Git tag keeps the preceding 56-condition publication.
This supplement extends its reports without changing any previous metrics or
predictions. See [Llama execution instructions](../../docs/llama_budget_extension.md).
