# Self-hosted inference cost scenarios — extension of #10

Inference cost is the primary economic comparison. Preparation and amortization
remain complementary work in #12. This extension reuses #10's usage/rate cards and
#11's recorded timing windows; it never calls a model to generate a cost report.

## What the report means

The original total and cost per 1,000 valid classifications remain unchanged.
They describe the benchmark's recorded attempts, including warmup and failures.
A separate `self_hosted` section describes **projections** for a local model on
one measured hardware allocation. Hosted API latency cannot be used to infer
self-hosted throughput and is explicitly unsupported here.

The continuous inference rate is:

```text
valid classifications/hour = valid evaluation outputs / evaluation window hours
USD/1,000 valid classifications = allocation USD/hour × 1,000 / valid classifications/hour
```

The evaluation window runs from the first call's start to the last call's end,
including failed-call time and inter-call gaps. It is not just the sum of model
compute time. Valid classification means a usable label, not a correct label.
A completed fixture's throughput is still only a fixture measurement: the report
shows sample count, latency, failure rate and source run metadata. Failed or
incomplete runs cannot establish the projection sample and remain unavailable.

For each requested successful volume `V` (default: 1,000 / 10,000 / 100,000):

| Scenario | Projected cost | Assumptions |
| --- | --- | --- |
| Continuous processing | `rate × (V / measured throughput + one warmup)` | Same measured workflow, no idle allocation after processing |
| Deployed for `H` hours | `rate × H` | One allocation stays provisioned; idle hours cost money |

Per-1,000 cost divides each scenario's cost by `V / 1,000`. Deployed demand is
**infeasible** when `V / throughput + warmup > H`: the allocation price remains
known, but cost per 1,000 completed outputs is unavailable. The report never
silently increases replica count or promises to finish excessive demand.

Warmup uses its complete measured time window and is counted **once** per
scenario. The continuous per-1,000 headline excludes warmup; each volume row
includes it. With no explicit warmup, projections are labeled unwarmed and may
include cold-call effects. Model construction, loading, fitting, and deployment
startup beyond measured warmup are excluded here and belong to preparation.

These are flat-rate cost projections, not provider invoices or capacity/SLA
guarantees. They assume the same inputs, batching, concurrency, cache conditions
and failure mix. They do not model traffic bursts, waiting for a batch to fill,
changing utilization efficiency, autoscaling, availability, billing minimums or
rounding. The busy fraction is projected occupied time, not measured GPU utilization.

## Hardware and pricing

A rate is for the **entire measured allocation**, not per core or GPU. Use a card
with source, currency, effective date, resource description and assumptions.
Matching device type alone does not establish equivalent performance.

Two explicit interpretations are supported:

- `cloud_equivalent`: an illustrative comparison using a declared reference rate.
  Actual runtime/hardware remains visible. The report never claims this is the
  reference cloud machine's measured throughput or an owned-machine invoice.
- `measured_hardware`: a rate for the actual allocation used by the measurement.
  Its `hardware_profile` must exactly match `--hardware-profile` recorded with the
  benchmark. Missing/different identities or devices leave prices unavailable.
  This is an operator declaration, not automatic verification of an instance or
  hardware inventory; use a profile that identifies instance/allocation, region,
  resource count and relevant configuration. Estimates are still not observed charges.

For a measured-hardware card, use this structure with your actual documented values:

```json
{
  "schema_version": 1,
  "currency": "USD",
  "api_rates": [],
  "local_hardware": {
    "interpretation": "measured_hardware",
    "hardware_profile": "YOUR-ALLOCATION-PROFILE",
    "device": "cpu",
    "resource": "Description of the entire measured allocation",
    "usd_per_hour": "REPLACE_WITH_DOCUMENTED_RATE",
    "source": "REPLACE_WITH_RATE_SOURCE",
    "effective_date": "YYYY-MM-DD",
    "assumptions": "Allocation size, region, pricing plan and exclusions"
  }
}
```

Replace the placeholders before use. A GPU run needs its own matching device,
profile and whole-allocation rate. No GPU rate or performance estimate is guessed.

[`self_hosted_reference.json`](../pricing/self_hosted_reference.json) supplies an
opt-in **illustrative** CPU rate for the two scenarios. It reuses the historical
September 16 price snapshot and explicitly defines the extrapolation assumptions.
It is not a fresh price verification or a recommended cloud deployment. Original
saved pricing and the original #10 ledger totals are not overwritten by repricing.

## Commands and recorded conditions

```bash
# Real local fixture measurement. Hardware-profile identity is optional for a
# cloud-equivalent rate; required when selecting a measured_hardware rate.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  venv/bin/python scripts/probe_latency.py \
  --run-id self-hosted --output-root /tmp/self-hosted-check \
  --hardware-profile local-cpu-fixture \
  --pricing pricing/self_hosted_reference.json

# Extrapolate a hypothetical 24-hour allocation; no new inference or deployment.
PYTHONPATH=src venv/bin/python scripts/report_costs.py \
  /tmp/self-hosted-check/self-hosted --deployment-hours 24 \
  --volumes 1000 10000 100000 \
  --output /tmp/self-hosted-costs.md --json-output /tmp/self-hosted-costs.json
```

Omit `--deployment-hours` to report only the continuous scenarios; no daily or
monthly schedule is assumed. `--pricing` on the report command selects another
rate card. It changes estimates, not throughput, original files or model outputs.

Both maintained campaign CLIs also accept `--hardware-profile`. New runs record
raw test-input Unicode-character length count/min/max/P50/P95 in `config.json`,
plus BERT's truncation limit or MiniLM's encoder length limit when available in
`measurement.json`. Character counts and model limits are not actual token-length
distributions; token lengths remain unavailable. Historical runs retain missing
length information rather than inventing it. Batch execution semantics, hardware,
thread settings, concurrency, cache conditions and original sample IDs remain in
the supporting artifacts. Reports hash timing/config evidence as well as the
existing usage, rate card and measurement metadata.

FLOPs per prediction are optional diagnostics. They remain unavailable until a
consistent operation-count method is introduced. Peak hardware TFLOPS do not
replace measured throughput or determine the dollar estimate.

## Validation

Deterministic tests use USD 2/hour, two valid classifications per evaluation hour,
and one hour of warmup. Four valid classifications take three hours including
warmup and cost USD 6 continuously. A four-hour deployed allocation costs USD 8,
including one idle hour. Eight valid classifications cannot fit in that allocation.
The fixtures also cover failed-call throughput, missing/mismatched prices, hosted
APIs, incomplete runs, zero successes, invalid scenario arguments, hardware profile
matching, input-length metadata and offline CLI repricing without artifact mutation.

A real local TF-IDF fixture validates recording/report generation; it does not
establish production performance or cloud hardware equivalence. Repeated short
texts and CPU-only measurements cannot justify GPU or general deployment claims.

Offline validation: **204 passed, 3 external integration tests deselected** using
`PYTHONPATH=src venv/bin/pytest -q -m 'not integration'`.

Recorded local evidence is under
[`artifacts/self_hosted_validation`](../artifacts/self_hosted_validation): a real
200-example TF-IDF fixture with five warmup examples, raw usage/timings, pricing,
hardware/input-length conditions and 24-hour allocation projections. The latter
are hypothetical; no resources were deployed and no paid requests were made.
