# Latency and throughput measurement — issue #11

## Contract

The runner measures each `classifier.predict(inputs)` call with
`perf_counter_ns`, from entry until the normalized prediction list is returned.
This includes classifier-specific prompt construction, tokenization, TF-IDF or
embedding extraction, device transfers, model/API inference, response parsing,
probability mapping, and output materialization. The built-in GPU adapters copy
outputs to CPU before returning, so their timed calls finish after device work.
Custom adapters must likewise return completed predictions, not pending work.

Dataset loading and classifier construction happen outside this boundary.
`prepare` and `fit` are timed separately; model loading inside those methods is
included in their stage times. They are excluded from inference. The runner
cannot recover constructor time from an already constructed classifier, so that
time is explicitly unavailable. More detailed fitting/loading stage attribution
belongs to issue #12.

Warmup is opt-in, using the first N **fit-training** examples and the configured
call batch size. Its IDs, durations, and failures are saved separately; it neither
uses test inputs nor contributes to quality, inference percentiles, or throughput.
The default is zero warmup to avoid adding implicit paid API calls. Such a run is
labeled **unwarmed; cold-start may be included**, not steady-state. After explicit
warmup, the report says `after_explicit_warmup`; it does not assert that a fixed
number of calls guarantees steady state or controls provider caches.

## Calls, batches and throughput

Default inference call batch size is one and concurrency is one. Both campaign
CLIs expose `--inference-batch-size`, `--warmup-examples`, `--client-location`, and
`--cache-condition`. The existing adapters loop sequentially inside `predict`;
passing four inputs is **not native tensor batching or four concurrent requests**.
Metadata records this execution mode. There is no concurrency scheduler in this
change.

- A **call latency** is the wall time of one `predict` invocation. Only when its
  batch size is one can it be interpreted as an individual logical-request latency.
- **Amortized time per example** is call duration / batch size. It describes a
  processing rate, not how long an individual request waited. Never calculate
  request-latency percentiles from amortized values replicated across examples.
- Report successful and failed/invalid call populations separately, including
  observation counts and P50/P95. Group successful calls by actual batch size,
  including the last partial batch.
- **P99 is unavailable below 1,000 observations in its population.** This rule
  leaves approximately ten observations in the upper 1%; it is a conservative
  reporting heuristic, not an uncertainty guarantee. The rule applies to
  `LatencyP99Metric` and regenerated reports. Percentiles use linear interpolation.
- Primary throughput is successful examples divided by wall time from the first
  evaluation call start to the last evaluation call end. Failed-call time and
  inter-call validation/logging are included. The report also provides active-call
  throughput, using the sum of call durations, to expose observer/inter-call
  overhead. Neither includes prepare, fit, warmup or final report serialization.
  Report attempted, successful and unattempted counts alongside throughput.

On a failed run, throughput describes only its observed prefix, not a completed
campaign. If a batched call fails or returns invalid outputs, none of its inputs
is counted as successful: the runner cannot reliably recover partial completion
from an exception. It preserves the timing and stops, retaining the existing
fail-fast behavior. Persisting/resuming partial predictions is still an issue #15
dependency; this change preserves individual timing observations.

## Artifacts and compatibility

Every non-dry runner execution that reaches preparation writes:

| File | Contents |
| --- | --- |
| `config.json` | Requested measurement settings and exact existing split/model configuration |
| `measurement.json` | Boundary, clock, planned test count, run state, batch/concurrency, warmup/cache/location, hardware/device, CPU affinity, threads, library versions, retry/timeouts |
| `timings.jsonl` | One durable row per preparation stage or inference call, monotonic start/end offsets, elapsed ms, input IDs, batch size, phase, success/exception/invalid-output state |
| `operational_report.json` | Aggregation of the two measurement files, including failed runs |
| `predictions.jsonl` | Successful completed-run predictions, with timing linkage as described below |

For singleton calls, new prediction `latency_ms` is the common outer-call time.
The previous adapter-specific time is preserved as `adapter_latency_ms`.
`latency_scope`, `inference_observation_index`, `inference_batch_size`, and
`amortized_latency_ms` make the relationship explicit. For batched calls,
`latency_ms` is null, so per-example latency metrics are unavailable; use the
call-level operational report instead. Quality/calibration outputs are unchanged.

Historical artifacts lack this boundary and remain adapter-specific observations.
Do not mix their latencies with the new outer-call population. Recomputing an old
artifact now adds P95 and suppresses unsupported P99; it does not repair or expand
its recorded timing boundary. No historical artifacts are overwritten here.

## Retries, hardware, and hosted APIs

The runner does not retry. Its default OpenAI client now explicitly uses
`max_retries=0` and timeout 120 seconds. The default Emissary client performs no
application retries and uses connect/read timeouts of 10/120 seconds. Reports
inspect injected OpenAI retry/timeout settings and Emissary session adapter retry
settings; unknown settings remain unknown. A caller injecting a retrying transport
gets logical-call timing including any backoff/retries, **not individual transport
attempt timings**. Those hidden timings are marked unavailable. Use zero-retry
clients for a one-attempt measurement. Requests that raise exceptions have their
own failed-call durations; exception type is retained without response text in
`timings.jsonl`.

Local hardware is the executing machine. For hosted APIs it is the **client**
machine; server hardware/queue time is not exposed. Location is caller-supplied,
not inferred from credentials/IP. Cache conditions are declarations, not verified
cache hits. Numeric-library versions, environment thread settings, CPU affinity,
actual local device, torch threads and initialized GPU names are recorded without
loading a model just to inspect hardware.

Hosted latency includes client preprocessing, transport/network, service queuing
and compute, and response parsing. Local latency includes preprocessing, local
compute, transfers and output materialization. A common client-visible boundary
makes the work included explicit; it does not make the underlying systems or
server-compute times identical. Record cache/device/location conditions when
comparing runs. Do not interpret the fixture smoke as representative production
latency or throughput.

## Reproduce a small local run

Both commands use real TF-IDF + LR on 200 repeated short fixture examples, with
five warmup examples. They require no credentials, downloads, or paid calls.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  venv/bin/python scripts/probe_latency.py --run-id single \
  --output-root /tmp/latency-check

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  venv/bin/python scripts/probe_latency.py --run-id batch4 \
  --inference-batch-size 4 --output-root /tmp/latency-check
```

Regenerate a human-readable report and JSON **without repeating inference**:

```bash
PYTHONPATH=src venv/bin/python scripts/report_operational.py \
  /tmp/latency-check/single --output /tmp/single-report.md \
  --json-output /tmp/single-report.json
```

Programmatic use: pass `MeasurementConfig(batch_size=1, warmup_examples=5, ...)`
from `llm_classifier_bench.measurement` as `BenchmarkRunConfig(measurement=...)`.
Warmup must fit within the actual fit-training partition. Campaign manifests
record the same settings; CSV/JSON summaries add P95, successful call count and
observed-window throughput.

Validation: 162 offline tests passed, three external integration tests deselected.
Deterministic-clock tests distinguish prepare/fit/warmup from inference, include
failed time in throughput, check output validation failures and batched timing,
and regenerate identical reports from saved observations. Additional tests cover
P99 call-count gating, corrupt observations, explicit SDK retry settings and both
maintained campaign CLIs. The local smoke reports and supporting observations
are recorded under `artifacts/latency_validation/`.

### Executed evidence

Both final local runs used committed code `4c78419` with a clean working tree,
200 fixture inputs, five warmup examples, one CPU thread requested through
OMP/OpenBLAS, and no network calls. Each generated Markdown and JSON report was
regenerated from saved observations and matched exactly.

| Run | Evaluation calls | P50 call ms | P95 call ms | Successful examples/s, observed window | P99 |
| --- | ---: | ---: | ---: | ---: | --- |
| [Single example](../artifacts/latency_validation/issue11-single/operational_report.md) | 200 | 0.933287 | 2.746468 | 705.996464 | unavailable |
| [Four examples per call](../artifacts/latency_validation/issue11-batch4/operational_report.md) | 50 | 3.673689 | 7.902935 | 887.794333 | unavailable |

Both completed all 200 examples without failures. The batch-4 median above is
the latency of a four-example call, not an individual example. These values
validate instrumentation on repeated short texts; they are not representative
performance claims or evidence that the four-example setting is generally faster.
Full configurations, predictions, fit metadata, raw timings and hardware/runtime
metadata accompany each report.
