# Inference usage and cost — issue #10

## Scope and outputs

Each non-dry benchmark run now saves:

- `usage.jsonl`: append-only inference attempts, runner call outcomes, and local timed calls.
- `pricing.json`: the exact selected USD rate card; omitted rates remain unavailable.
- `cost_report.json`: costs, coverage, failures, per-sample allocations and evidence hashes.
- Existing `measurement.json` and `timings.jsonl`: actual runtime/device and timing evidence.

`predictions.jsonl` gains `cost_usd`, `cost_kind`, and `usage_event_ids`. These are
**evaluation costs for that sample**. The authoritative run total also includes
warmup and failed attempts; summing successful prediction rows loses those costs.
`evaluate_jsonl` replays the adjacent ledger for the canonical `predictions.jsonl`
and verifies the successful sample IDs match. Historical files without a ledger
retain their original per-record metrics behavior. Copy the complete run directory
when reproducing new cost metrics.

`total_cost_usd` covers all recorded inference attempts, including warmup, invalid
responses and failures. `cost_per_1000_usd` in metrics is:

```text
1000 × total inference cost / successfully classified evaluation examples
```

Successful means a valid accepted classification, not necessarily a correct label.
Warmup never enters this denominator. Failure rate is unsuccessful / submitted
evaluation examples. The runner still stops at the first failure. If a batch fails,
none of its outputs are accepted; the submitted batch counts as unsuccessful even
if a sequential adapter processed only part of it. Only actual recorded API
attempts are priced. Use batch size 1 for exact per-request failure attribution.
Failed runs retain their usage/cost reports even though the final prediction file
may never be written. Preparation is excluded and belongs to issue #12.

## Usage, retries and missing information

OpenAI and Emissary adapters emit one event per application-level API attempt,
including the requested model, returned model, request ID, numeric usage fields,
cache counts when exposed, phase, outcome and transport retry coverage. Usage is
captured **before parsing the label**. No response probabilities are invented;
OpenAI probability-based calibration metrics remain unavailable.

Default clients use zero automatic transport retries. If an injected client can
retry internally, its final response does not account for every potential charge:
we retain its known subtotal and mark total coverage unavailable. The ledger can
sum separately recorded billed retries for the same sample without duplicating
the successful-example denominator. This change does not add a retry loop.

A timeout or error without billing units is **unknown**, even if the provider might
ultimately charge zero. Unknown models, service tiers, prices, unsupported audio,
cache-write charges, or malformed usage also remain unavailable. Emissary has no
verified inference tariff in the included card, so its cost stays unavailable.
An optional `observed_charge` ledger field requires an amount, USD currency and
source (e.g. a provider invoice line); it replaces an estimate for that event.
Current adapters do not infer invoice charges from undocumented response fields.

Reports distinguish `observed`, `estimated`, `mixed`, `unavailable`, and
`no_attempts`. A zero known subtotal is **not** a complete zero bill when coverage
is unknown. Missing call usage, unmatched attempts and interrupted runs without a
final measurement status also prevent complete totals. Append-only logs survive
ordinary exceptions; they cannot recover usage lost in a process kill or an
unobserved provider retry. They are not a billing reconciliation service.

API usage persistence is inside the adapter call and therefore contributes a small
amount to its measured client latency. Local timing is converted to cost after the
call, so reporting work is not charged as local inference.

## Versioned example rates

The opt-in card [`inference_2026-09-16.json`](../pricing/inference_2026-09-16.json)
records source, effective date, currency and assumptions. The date means **verified
on that date**, not the provider's original tariff start date. Explicitly selecting
this card means repricing recorded use under these assumptions, not reconstructing
an old invoice. No rate is silently inferred for an unknown model or tier.

### OpenAI standard text inference

The [official GPT-5 nano model page](https://developers.openai.com/api/docs/models/gpt-5-nano)
lists USD 0.05 / 0.005 / 0.40 per million input / cached input / output tokens.
The card explicitly maps `gpt-5-nano` and `gpt-5-nano-2025-08-07`, standard `default`
service tier. No Batch, Flex, priority, audio, tool, tax, discount or regional
surcharge assumptions are included.

[Cached input](https://developers.openai.com/api/docs/guides/prompt-caching) is a
subset of input tokens: subtract it before applying the ordinary input rate.
[Reasoning tokens](https://developers.openai.com/api/docs/guides/reasoning) are
already included in completion tokens and are not added a second time. Missing
cached-token counts prevent a complete estimate. Nonzero separately billed cache
writes require a corresponding rate; the example card has none.

### Local compute

The example CPU rate is an **illustrative cloud-equivalent estimate**, not an
owned-hardware bill or a claim of equal performance. The
[AWS Lightsail Linux public-IPv4 bundle table](https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-bundles.html)
lists 8 vCPUs / 32 GB at USD 164/month. Dividing by an **assumed 730 hours/month**
gives USD 0.2246575342/hour. This is a monthly-price allocation, not a quoted hourly
offer. We apply it only to active measured inference wall time:

```text
cost = active inference milliseconds / 3,600,000 × selected USD/hour
```

Actual machine details, device, affinity and thread settings remain in
`measurement.json`; the selected rate's resource description remains in
`pricing.json`. Matching `device=cpu` prevents accidentally applying a CPU rate to
CUDA/MPS, but does **not** establish performance equivalence. Select an appropriate
rate for the experiment before comparing methods. No GPU rate is supplied. Idle
provisioning, setup, total allocation time, energy and depreciation are excluded.
Local batch cost is counted once and divided equally across its accepted examples.

## Run and reprice

Both maintained Banking77 campaign scripts and `probe_latency.py` accept:

```bash
--pricing pricing/inference_2026-09-16.json
```

Leaving this flag out still records usage, but supplies no price assumptions.
Python callers use `BenchmarkRunConfig(pricing_path=Path(...))`. Rates are validated
before classifier preparation or inference.

```bash
# Regenerate from the exact saved pricing and usage; no inference/network.
PYTHONPATH=src venv/bin/python scripts/report_costs.py artifacts/runs/RUN_ID

# Compare a different card without modifying the original run.
PYTHONPATH=src venv/bin/python scripts/report_costs.py artifacts/runs/RUN_ID \
  --pricing path/to/alternative-pricing.json \
  --output artifacts/repricing/RUN_ID.md \
  --json-output artifacts/repricing/RUN_ID.json

# Small offline validation: simulated API usage + real TF-IDF CPU inference.
PYTHONPATH=src venv/bin/python scripts/probe_costs.py \
  --output-root artifacts/cost_validation/issue10
```

The replay command refuses to overwrite existing files inside the source run.
Repricing produces an alternative report; it does not rewrite saved predictions
or their original metrics. Reports include usage and selected-pricing hashes.

## Known-usage validation

The probe uses the real OpenAI adapter/runner with an injected deterministic client;
**all API responses and token counts are simulated, with no provider requests or
charges**. Its local TF-IDF run executes actual CPU inference. These checks validate
accounting, not model quality, representative performance or provider invoices.

| Case | Independent expectation |
| --- | --- |
| One simulated response: 1,000 input, 400 cached, 100 output including 20 reasoning | `(600×0.05 + 400×0.005 + 100×0.40)/1e6 = USD 0.000072` |
| Three accepted predictions plus one warmup | USD 0.000288 total; USD 0.096 / 1,000 successes |
| One accepted response plus invalid-label response using 200 uncached input / 10 output | USD 0.000086 total; USD 0.086 / 1,000 successes; failure fraction 0.5 |
| Local TF-IDF including warmup | Sum saved inference durations × documented illustrative hourly rate |

Evidence is saved under [`artifacts/cost_validation/issue10`](../artifacts/cost_validation/issue10).
Tests additionally cover partial failed batches, unknown usage, hidden retries,
explicit billed retry events, observed charges, wrong-device rates, invalid prices,
independent repricing, legacy metrics compatibility and sample identity checks.

Offline regression result: **192 passed, 3 integration tests deselected** with
`PYTHONPATH=src venv/bin/pytest -q -m 'not integration'`. No paid API validation
was performed. The evidence records the committed code revision and whether its
working tree was clean when the probe started.
