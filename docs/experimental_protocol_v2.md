# Experimental protocol and interpreting results (v1.3)

The benchmark compares classification quality, calibration, latency and cost.
A result describes a particular dataset, label set, supervision budget, model and
execution environment. It is not a general ranking of model families.

## Comparison regimes

Use one regime per campaign and retain it in every table or plot:

- **Full-training reference:** local supervised methods use the sampled training
  data; OpenAI and Jev are zero-shot; Emissary uses its configured shot budget. These
  methods do not necessarily consume equal amounts of labeled information.
- **Matched labeled budget:** each method receives the same selected fit/context
  pool. Training examples and in-context demonstrations are different uses of
  the same label budget. An additional validation budget is recorded separately.

For matched runs, compare the same dataset, class count, seed, pool hash, test IDs
and frozen class definitions. Report both allocated labels and actual consumption.
At nonzero validation budgets, local supervised methods consume validation labels
that OpenAI and Emissary do not; equal fit budgets do not imply equal total label
consumption. Jev supports zero fit and validation budgets only; other matched
cells are explicitly unsupported. See [matched budgets](matched_label_budgets.md) for the supported matrix.

Emissary routing and Projects SFT use different mechanisms and potentially different
base models. Changing shots between those products does not isolate the effect of
additional labels. See the [adapter contract](emissary_contract.md).

## Data selection and reproducibility

The original source test split is reserved for final evaluation. Only source
training examples are partitioned into fit and validation. Do not select model
settings or edit class descriptions in response to test results.

Banking77 campaigns sample nested label prefixes for each seed. Source examples
are sampled independently per label. Test IDs remain paired across methods and
budgets within a condition. Fit/validation membership can change when the class
count changes, so cross-class-count comparisons do not guarantee identical fit
examples for shared labels. Check saved IDs before comparing a fixed test cohort.

The v2 campaign can reduce requested source support when too few labels qualify.
`campaign.json` records requested and effective support. Use `--strict-support`
when the requested design must be preserved exactly. Matched budgets are selected
from the fit/validation partitions after this source sampling and splitting.

Overlapping sample IDs or exact UTF-8 text across fit, validation and test are
rejected. This is not semantic deduplication. Freeze any data cleanup before
running models; do not silently resample a failed condition.

Retain the following when sharing a result:

- Code revision and all CLI options, including seeds and requested/effective budgets.
- Original source data or an immutable source revision and content hashes.
- Frozen definition profile, its review status, and SHA-256.
- Exact fit, validation and test IDs from `config.json`; matched pool identity
  and consumption from `labeled_budget.json` when applicable.
- Requested/resolved model identifiers, training settings and `fit_metadata.json`.
- Dependency versions, device, thread settings, location and cache declarations.
- Predictions, status, raw timing/usage/preparation records, and selected pricing.

Seeds alone are insufficient: dataset IDs are derived from source row positions,
some loaders use mutable source URLs, and remote services may be nondeterministic.
The checked-in definition profiles are inputs, not evidence of a completed benchmark;
inspect their review metadata before selecting a final profile. See
[class definitions](class_definitions.md) and the README's saved-settings replay example.

## Model selection

TF-IDF and SentenceTransformer LR select the highest validation accuracy; ties
retain the first configured C. This criterion need not select the best probability
calibration. BERT restores the epoch with the lowest mean validation loss per
example; exact ties retain the earlier epoch. No train-plus-validation refit is
introduced after selection.

Without validation, TF-IDF uses its fixed fallback C, SentenceTransformer LR
retains its first candidate, and BERT retains its final epoch. Report that policy
alongside the chosen settings. Never select C or an epoch using final test scores.

## Reading quality and calibration

| Metric | Interpretation |
| --- | --- |
| Accuracy | Fraction of correct labels; higher is better |
| Macro-F1 | Mean class F1; higher is better. By default, the label set is the union of gold and predicted labels present in the evaluated records |
| Multiclass log loss | Mean negative natural log probability of the gold class, clipped with epsilon `1e-15`; lower is better |
| Multiclass Brier score | Squared probability errors summed over classes and averaged over examples; lower is better |
| Top-label ECE | Weighted confidence/accuracy gap over fixed-width confidence bins; lower is better |
| Adaptive ECE | Approximately equal-frequency bins that keep identical confidences together; lower is better |

Default ECE uses 10 requested bins. Adaptive ECE can produce fewer bins when
confidences tie. Its `equal_frequency_preserve_ties_v2` metadata identifies the
policy; do not mix it silently with results calculated using a different policy.
Bin-based ECE is sensitive to sample size and binning. Accuracy and calibration
measure different properties; report both.

Jev, Emissary, runner validation and default metrics accept probability sums within an
absolute `1e-4` of one without renormalizing the values. Probabilities must be
finite and in [0,1], and consistent with the predicted label and confidence.
Duplicate evaluation IDs are rejected.

Jev's benchmark confidence is the probability assigned to its selected label.
Its native distribution-concentration score is stored separately and is never used
as top-label confidence. Exact probability ties accept the provider's selected
maximum regardless of mapping order; no random tie breaking is introduced. The
pinned Jev configuration is `jev-1.13.0`, one Choice per text, at most 255 classes,
with a conservative preflight against a 32,000-token single-question budget. Limits,
requested/resolved versions, retries and request hashes are retained; changing an
alias or the operator's context limit requires documenting that change. See
[Jev configuration and limits](jev.md) for the dated provider sources.

OpenAI returns labels without probabilities or confidence, so its calibration
metrics are unavailable. Missing metrics and unknown costs are not zero. Check
`available` and the associated reason in each result.

## Failures and incomplete comparisons

`status.json` distinguishes completed, failed, unsupported and planned dry runs.
Report all requested conditions and reasons for missing results. Do not silently
average away failed seeds or unsupported methods.

The runner stops on an exception or invalid output. Earlier validated batches
remain in `predictions.jsonl`; a failed batch has no recoverable partial output
under the classifier interface. Ctrl-C is recorded as a failure and propagated.
The default batch size of one preserves each earlier successful prediction.

No complete-run quality metrics are written for failed runs. Reevaluation of a
runner artifact with a usage ledger requires a completed measurement and full
recorded test coverage. Automatic resume and aggregate quality metrics that count
failed predictions as errors are not implemented. Partial timings and costs remain
useful operational evidence, with their coverage reported explicitly. Jev checks
label/context limits and matched-budget support before network access, preserving
unsupported conditions in campaign summaries.

## Latency, cost and uncertainty

Compare the same input distribution, batching, warmup and hardware/cache conditions.
Individual call latency differs from amortized time per example. Hosted latency
includes network and provider waiting; local latency includes local preprocessing
and compute. P99 requires at least 1,000 observations. See
[latency measurement](latency_measurement.md).

Inference cost includes recorded warmup and failed attempts. Reports distinguish
observed, estimated and unavailable values. Costs per successful prediction use
valid outputs, which need not be correct. Rate cards are dated assumptions, not
proof of an invoice. [Self-hosted projections](self_hosted_inference_costs.md) and
[preparation amortization](preparation_costs.md) have separate scopes and assumptions.

The campaign saves per-run results; it does not calculate cross-seed confidence
intervals or bootstrap estimates. Publish seed-level results and prediction
coverage. For statistical comparisons, predeclare the aggregation and uncertainty
method and pair observations by test ID; repeated or overlapping test cohorts are
not independent datasets. Small smoke runs only verify execution. The Jev pilot caps paid attempts at four,
uses a fresh local baseline on the same held-out IDs, and must be labeled live
smoke evidence; synthetic transport fixtures are separate offline evidence. Reuse
older baselines only after verifying source/text identity, frozen definitions,
label order, splits, budgets and measurement configuration. Otherwise record the
mismatch and rerun the affected baseline under an explicitly chosen budget.

## Inspect and recalculate your results

Run these commands from the repository root with the environment activated:

```bash
PYTHONPATH=src python scripts/evaluate_artifact.py \
  artifacts/benchmark_runs/CAMPAIGN/runs/RUN/predictions.jsonl \
  --output /tmp/recalculated-metrics.json

PYTHONPATH=src python scripts/audit_calibration.py \
  --campaign artifacts/benchmark_runs/CAMPAIGN --output /tmp/calibration-audit.json

PYTHONPATH=src python scripts/report_operational.py \
  artifacts/benchmark_runs/CAMPAIGN/runs/RUN --output /tmp/operational.md

PYTHONPATH=src python scripts/report_costs.py \
  artifacts/benchmark_runs/CAMPAIGN/runs/RUN --output /tmp/costs.md
```

Recalculation uses saved evidence without repeating model inference. The calibration
audit compares saved values with the current metric implementation and independent
NumPy formulas; a policy change can legitimately make an old metric comparison fail.
Use a separate output path and keep the original report when comparing versions.
Generated run directories are excluded from source control by default. Share a
complete selected campaign separately rather than treating local smoke outputs
as release results.
