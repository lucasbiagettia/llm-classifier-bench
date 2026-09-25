# Jev setup and result interpretation (v1.3)

Jev is an opt-in zero-shot classifier in both maintained Banking77 campaigns.
It uses the [TypeSafe Choice API](https://docs.typesafe.ai/primitives/choice),
sending one text and one question per request. Class names and frozen descriptions
become the criteria without rewriting, shortening or adding labels. Training and
validation consumption are both zero. Nonzero matched fit **or validation** budgets
produce an `unsupported` cell before API access.

## Setup and a bounded pilot

Set `JEV_TOKEN` in the environment or the repository's ignored `.env` file.
The adapter uses the existing `requests` dependency. No provider SDK is needed.
Run from the repository root with the Python environment activated:

```bash
# Plan only: data loading is allowed, but no inference or credentials are used.
PYTHONPATH=src python scripts/probe_jev_classifier.py

# Paid: exactly one small condition, at most FOUR Jev attempts, no retries/warmup.
# Also runs the local TF-IDF baseline on identical held-out IDs.
PYTHONPATH=src python scripts/probe_jev_classifier.py --live
```

The pilot selects two Banking77 classes with seed 42, eight source-train examples
and two source-test examples per class, then reserves 25% of training for validation.
It uses the existing frozen enriched definition profile, whose review status is
currently `unreviewed`. This is execution evidence, not a formal benchmark result.
Full-training-reference supervision differs: TF-IDF consumes fit/validation labels;
Jev consumes none. A new invocation creates a new run and can spend again.

For a custom campaign, both entry points accept:

```bash
PYTHONPATH=src python scripts/run_banking77_scaling_benchmark_v2.py \
  --classifiers jev --class-counts 2 --seeds 42 \
  --train-per-class 8 --test-per-class 2 \
  --min-train-per-class 8 --min-test-per-class 2 --strict-support \
  --jev-model jev-1.13.0 --jev-max-requests 4 --jev-max-retries 0 \
  --pricing pricing/inference_2026-09-25.json
```

`--jev-max-requests` is a **per-run attempt cap**, including warmup and retries;
it is not a dollar ceiling or a whole-campaign limit. The dedicated probe fixes
one run and four attempts. Campaign defaults leave this cap unset. Explicit options
also include `--jev-endpoint`, `--jev-timeout-s`, `--jev-retry-backoff-s` and
`--jev-context-window-tokens`. HTTPS endpoints must contain no credentials/query;
redirects are disabled. Injected transports have unknown hidden-retry coverage.

Retries default to zero. When enabled, transport errors, 429, 529 and selected 5xx
statuses use bounded exponential backoff. Every attempt enters `usage.jsonl`;
invalid outputs and other HTTP errors are not retried. Exceptions omit response
bodies and credential-bearing transport details. A timeout can have unknown cost.

## Supported contract

The pinned default is `jev-1.13.0`. Requested model, returned model, request hash,
request ID when exposed, attempt count, endpoint and numeric usage are retained
in the run artifacts. Aliases remain configurable; inspect every returned model
before aggregating results. A returned version without an exact price match has
unavailable cost. The API credential and authorization headers are never stored in result artifacts;
numeric usage counts are retained.

The [model documentation](https://docs.typesafe.ai/models), checked 2026-09-25,
lists 64k tokens per request and 32k for state plus the longest question. The single
question adapter defaults to 32,000. Choice supports at most 255 labels. Preflight
uses serialized UTF-8 bytes plus 1,024 units of framing as a conservative token
estimate, not the provider tokenizer. Oversized inputs are unsupported; no silent
truncation. These documented limits describe the pinned model, not arbitrary
future models; an operator changing versions must verify and configure its limits.

A response must include the resolved model, Choice answer, configured label and
complete label-to-probability mapping. Unknown/missing labels, nonfinite values,
values outside [0,1], sums outside absolute `1e-4`, or a nonmaximum selected class
fail explicitly. Values are not renormalized. Exact ties retain the provider's
chosen maximum, independent of JSON key order; tied labels are saved.

Benchmark `confidence` equals the selected class's probability. Native
[TypeSafe confidence](https://docs.typesafe.ai/confidence) describes concentration
and is retained separately as `raw_response.provider_confidence`; it never feeds
ECE. Missing native confidence is identified without discarding valid probabilities.
Missing usage leaves cost unavailable. The normal metric pipeline supplies accuracy,
macro-F1, both ECE variants, log loss, Brier, latency and inference cost.

## Costs and replay

The dated card adds the published standard price of USD 0.042 per million input
tokens for the pinned model at the public endpoint; output is free. Estimates use
reported `input_tokens`, not the preflight estimate. Other rates in this combined
card retain their earlier verification dates and assumptions. Local CPU costing is
an illustrative cloud equivalent, not a measured electricity bill. Remote preparation
makes no calls and is recorded as no billable preparation work.

Keep the complete generated campaign: `summary.json`/CSV, definitions, each run's
config/status, predictions, timings, usage, pricing and cost/preparation reports.
Recalculate without any API access:

```bash
PYTHONPATH=src python scripts/evaluate_artifact.py \
  artifacts/jev_pilot/CAMPAIGN/runs/RUN/predictions.jsonl --output /tmp/jev-metrics.json
PYTHONPATH=src python scripts/report_costs.py artifacts/jev_pilot/CAMPAIGN/runs/RUN
PYTHONPATH=src python scripts/report_operational.py artifacts/jev_pilot/CAMPAIGN/runs/RUN
```

Fixtures under `tests/fixtures/jev` are **simulated**; live runs live under ignored
`artifacts/`. Reuse a baseline artifact only when dataset/text content, label order,
definition hash, split/test IDs, comparison regime and measurement settings match.
No compatible baseline is shipped; the pilot runs a fresh local reference instead
of repeating other paid APIs. Four observations cannot establish a ranking,
calibration, tail latency or statistical significance. Publish failed/unsupported
cells and per-seed coverage; the existing runner does not compute confidence intervals.
