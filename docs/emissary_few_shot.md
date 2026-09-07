# Configurable Emissary labeled-example planning

**Nonzero live support is blocked by the public API contract.** See the
[contract investigation](emissary_contract.md). Zero-shot remains available;
nonzero conditions can be configured, selected, audited, and dry-run offline.
No examples are inserted into class descriptions or inference prompts.

`EmissaryTrainingConfig(shots=5, shot_unit="total", selection_seed=42,
selection_policy="balanced_round_robin_v1")` configures a condition.
`shots` is any nonnegative integer. Nonzero budgets require an explicit `total`
or `per_class` unit, including in dry runs. The examples below use **total** as an
implementation example; Tanmay's intended meaning of 5/100 is not confirmed.

The policy sorts class names and per-class sample IDs before seeded shuffles.
It shuffles classes using seed string `<seed>:classes`, and each class using
`<seed>:examples:<label>` with Python's `random.Random`. Round-robin ordering
assigns N // K items per class and gives the first N % K shuffled classes one
extra. Per-class N is the same prefix of length N*K. With fixed data, labels and
seed, smaller budgets are prefixes of larger ones, independently of input order.
A class with insufficient support fails its assigned quota; there is no budget
redistribution. Five total examples over twenty classes cover five classes.
This is locally valid planning, not a claim that the provider permits it.

Only the runner's fit-training partition is eligible. Validation and test are
never selected. Repeated IDs and exact UTF-8 text across partitions are rejected
before remote operations. Text duplicates within a partition remain separate
source examples; this is not fuzzy/semantic deduplication. Existing campaign
label eligibility and frozen definition profiles are preserved. In v2,
`--strict-support` prevents its existing source-sampling backoff; that backoff
never changes the separately specified Emissary shot budget.

## Commands

Activate the existing environment and run the regression suite:

```bash
source venv/bin/activate
PYTHONPATH=src pytest -m "not integration"
PYTHONPATH=src pytest tests/classifiers/test_emissary_few_shot.py tests/test_emissary_campaign.py
```

Dry-run a tiny Banking77 comparison with reusable local class definitions:

```bash
PYTHONPATH=src python scripts/run_banking77_scaling_benchmark_v2.py \
  --classifiers emissary --class-counts 2 --seeds 42 \
  --train-per-class 70 --test-per-class 1 \
  --min-train-per-class 70 --min-test-per-class 1 --strict-support \
  --validation-fraction .2 --emissary-shots 0 5 100 \
  --emissary-shot-unit total --emissary-selection-seed 42 \
  --definitions class_definitions_data/banking77/canonical_llm_enriched_v1.json \
  --dry-run --output-root artifacts/emissary_few_shot
```

The original `scripts/run_banking77_scaling_benchmark.py` accepts the same options
except v2's `--min-*-per-class` and `--strict-support`; omit those three options.
Both support `--emissary-shot-unit per_class` and arbitrary feasible budgets.
No credentials or definition generation are needed for dry runs. Dataset loading
can need network access/cache; the offline suite uses local fixtures. The exact
cached real-data command is in [validation evidence](emissary_validation.md).

`--emissary-shots 0` preserves the default zero-shot condition. Omitting
`--dry-run` executes that existing live path and requires credentials. A mixed
live invocation containing any nonzero budget is rejected before loading data or
creating clients, including its zero-shot condition. Dry runs skip other
classifier families and print/save only Emissary preparation plans (v2 retains
its existing label-only dry run when Emissary is absent).

## Artifacts and operational evidence

Microsecond campaign IDs plus condition names keep runs distinct; existing output
folders are never overwritten. All budgets share the condition's test IDs and
exact definition profile. Each run saves:

- `config.json`: pre-run settings, classes, fit/validation/test IDs and source
  profile hash. `classifier.training` contains the shot configuration.
- `preparation_plan.json`: requested/actual totals, all per-class counts, coverage,
  seed, policy, shuffled class order, and selected IDs/labels/text/order/content
  SHA-256. Includes supported operations, explicit live blocker, and unknown cost.
- `status.json`: `dry_run`/`planned`, or normal run completion/failure. Dry runs
  create no predictions, metrics or fit metadata; runner result `example_count=0`.
- `fit_metadata.json` after successful preparation/fit: pinned experiment/model
  version, actual zero-shot selection, experiment creation and preparation timing,
  null upload/training/job fields with reasons, raw creation response and unavailable
  cost. This is saved before inference, including when inference later fails.
- Live zero-shot predictions retain per-request latency and full provider responses
  (including usage/charge evidence if returned). Existing cost metrics remain
  unavailable when `cost_usd` is absent; unknown cost is never set to zero.

`plan_fit` is an optional generic runner hook receiving classes, fit examples and
validation examples, without test data. It must perform no remote operations.
Dry-run mode requires this hook. The lifecycle remains prepare → fit → predict.

## Bounded live smoke: prepared, blocked

[emissary_smoke_plan.json](emissary_smoke_plan.json) records the tiny 0/5/100-total
comparison, six maximum inference requests and a proposed USD 1 ceiling. It is a
reviewable plan, not an implemented provider spending control or confirmation of
Tanmay's unit. Do not turn it into a full benchmark.

The exact candidate live command is the dry-run command above with `--dry-run`
removed. **It currently exits before remote operations** because nonzero shots
are unsupported. It must remain blocked until the provider contract and pricing
permit enforcing the planned cap, and the intended shot unit is confirmed.
There is no honest executable nonzero live smoke command yet. The existing live
integration test would only exercise zero-shot and would not resolve acceptance.
