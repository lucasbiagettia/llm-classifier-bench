# Preparation investment and amortization

Inference remains the primary cost comparison. Preparation is now recorded as a
separate incremental investment, with offline amortization over 1,000, 10,000 and
100,000 **valid classifications**. This implementation does not change data splits,
fit examples, candidate grids, model-selection rules or final refit behavior.

## Evidence and timing boundaries

Non-dry runs now save:

- `preparation.jsonl`: individual stage intervals, parent IDs, outcomes and provenance.
- `preparation_metadata.json`: hardware/device, software/thread conditions, adapter
  resource IDs, billing availability and measurement exclusions.
- `preparation_report.json`: total time/cost, exclusive stage components, selected
  fit contribution, configuration/rate hashes and cache evidence.

`config.json` retains the exact data IDs, seed, training parameters and hardware
profile. `fit_metadata.json` identifies the chosen configuration. Fine-grained
records are emitted through the optional `set_preparation_sink` hook; unsupported
custom adapters still receive outer prepare/fit measurements.

The charged runtime estimate uses the **outer prepare and fit intervals once**.
Child stages are diagnostic components. Each event's exclusive time subtracts its
direct children before categories are summed. This prevents charging both a full
`fit` and every step inside it. The `parent` category is otherwise unattributed
setup/import/recording overhead, not another charge. Selection costs are visible
alongside all candidate fitting costs; the total investment includes both.

Warmup belongs to inference and never enters preparation. Classifier construction
before the runner, dataset loading and gaps between prepare/fit are outside the
active-wall-time estimate. Model loading within fit is included. When actual
billable allocation evidence includes idle time or billing rounding, that evidence
replaces the active-time estimate.

| Adapter | Measured stages and selection |
| --- | --- |
| TF-IDF + LR | Train-only vectorizer fit/transform, validation transform, each LR fit and validation score, selection |
| MiniLM + LR | Encoder loading/reuse, train/validation embeddings, each LR fit/score, selection |
| BERT | Model/tokenizer load, each epoch's training, nested tokenization, validation and its tokenization, checkpoint copy/restore, selection |
| Emissary | Experiment creation, dataset upload/profile, training submission/wait, deployment submission/wait; resource/continuation IDs retained |
| OpenAI | Local prepare/fit boundaries; no remote preparation calls in the zero-shot adapter |

CUDA BERT stages synchronize around measured work so asynchronous launches are not
mistaken for completed training. These synchronization points preserve the training
operations but can change scheduling overhead. Recording also adds overhead, visible
in parent intervals. Real CUDA performance was not measured by the offline validation.

TF-IDF and MiniLM retain the winning fitted candidate; **no final refit is added**.
`selected_fit_cost_estimate_usd` is a subset of total investment and excludes shared
features, validation, loading and restoration. It must never be added to the total.
For BERT, the selected fit is the **cumulative training prefix** through the chosen
epoch, not an independent candidate trained from scratch. Its exclusive fit time
excludes separately measured tokenization. Remaining epochs still count in total
investment because that work was actually executed.

## Cached work and provenance

Feature/embedding extraction is marked `computed_this_run`; these adapters do not
reuse a persisted feature cache. An injected or previously loaded MiniLM encoder
is marked `reused_in_memory`, with declared model, encoder type and checkpoint
revision when exposed. Model-loader disk-cache hits are **unknown**, not assumed
cold. BERT records its resolved checkpoint revision when exposed. Emissary retains
dataset payload hashes and continuation dataset/job/deployment identifiers.

Reports represent work paid/executed **in this run**. They do not charge an old
artifact again or infer that creating it originally was free. Prior artifact
creation cost remains unavailable. To compare preparation from scratch against
reuse, execute separate declared conditions; the reporter does not invent a cold
run or clear model caches. CPU affinity, threads, device and GPU inventory are
recorded; RAM quotas and GPU sharing/reservation remain unavailable when not
exposed. Hardware-profile matching is an operator declaration, not cloud inventory
verification.

## Money and provider evidence

Default local estimate:

```text
preparation USD = (prepare wall ms + fit wall ms) / 3,600,000 × whole-allocation USD/hour
```

Use the [inference rate-card conventions](inference_costs.md): USD currency, source, effective date, resource
description and assumptions. `measured_hardware` rates require a matching device
and recorded hardware profile. `cloud_equivalent` rates are explicitly illustrative,
not owned-machine bills or claims of equivalent cloud performance. The whole
allocation is assumed retained through each measured call, including CPU stages
on an allocation containing a GPU. Unknown rates remain unavailable.

Remote queue/wait durations are **never priced as local compute**. Current Emissary
responses expose no documented preparation charges or billable resource duration,
so cost remains unavailable. Reusing an existing routing model performs no remote
preparation and has zero *incremental provider preparation work*; its original
creation cost remains unknown. OpenAI zero-shot also performs no remote preparation.
Client-side construction/compute is excluded from these provider-charge statements.

An adapter can expose documented billing through `preparation_metadata()['billing']`.
Alternatively, pass `--billing evidence.json` to import a charge or a resource meter
without repeating preparation. The evidence must cover **entire preparation**;
partial line items are not silently presented as a complete bill.

Observed-charge example (synthetic format illustration, not an actual invoice):

```json
{"kind":"observed_charge","scope":"entire_preparation","source":"invoice/job reference","currency":"USD","amount":"12.00"}
```

Billable-resource example (hours must come from actual allocation evidence):

```json
{"kind":"billable_allocation","scope":"entire_preparation","source":"allocation meter reference","device":"cuda","hardware_profile":"YOUR-ALLOCATION","billable_hours":"0.5"}
```

Billable hours require a matching `measured_hardware` rate and refer to the entire
allocation; apply any billable minimum/rounding before recording those hours.
The reporter estimates the charge using that rate; it does not derive hours from
remote job wait. Evidence replaces the runtime estimate rather than adding to it.
Stage/selected-fit runtime estimates remain separate diagnostics and do not allocate
an observed invoice among components. Provider tariffs, credits and meter authenticity
are not independently verified by this offline reporter.

## Amortization and commands

Choose the inference scenario explicitly:

- `continuous`: the inference volume projection, including one measured inference warmup.
- `deployed`: the declared allocation-hours price, including idle time; infeasible
  volumes remain unavailable. `--deployment-hours` is required.
- `recorded_api`: recorded evaluation spend / valid outputs × volume, plus recorded
  warmup once. Requires complete hosted API cost coverage and a completed run.

```text
amortized USD/valid prediction = preparation USD / volume + inference scenario USD / volume
```

```bash
# New local fixture: no model download, credentials or paid calls.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/probe_preparation.py --output-root /tmp/preparation-check

# Offline continuous projection; optional --pricing selects another rate card.
PYTHONPATH=src python scripts/report_preparation.py \
  /tmp/preparation-check/local-tfidf --inference-scenario continuous \
  --output /tmp/preparation-continuous.md --json-output /tmp/preparation-continuous.json

# Offline deployed projection; this command does not deploy anything.
PYTHONPATH=src python scripts/report_preparation.py \
  /tmp/preparation-check/local-tfidf --inference-scenario deployed \
  --deployment-hours 24 --volumes 1000 10000 100000 \
  --output /tmp/preparation-deployed.md --json-output /tmp/preparation-deployed.json
```

Preparation and inference components remain visible even when their sum is unknown.
An incomplete preparation cannot be amortized as a deployable model's investment;
its measured time/spending is still retained. The runner preserves stages on ordinary
exceptions and detaches the recorder afterwards. Abrupt process termination can leave
missing parent records; inconsistent/incomplete evidence is not repaired by guessing.

Historical runs without preparation stage records require new measurements for
this metric. They still support their existing inference reports. No historical
preparation cost is reconstructed from inference latency or current machine speed.
Original evidence is never overwritten by the report CLI.
