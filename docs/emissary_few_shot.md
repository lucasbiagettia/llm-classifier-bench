# Configurable Emissary labeled-example support

Nonzero Emissary conditions use the documented **Projects classification
fine-tuning** flow. They do not add examples to, or retrain, a routing experiment.
The zero-shot condition continues to use an Emissary routing experiment.
Consequently, a 0-versus-5/100 comparison also changes mechanism and base model;
report that confound with any result.

`EmissaryTrainingConfig` supports arbitrary feasible nonnegative budgets. A
nonzero budget requires an explicit `shot_unit` (`total` or `per_class`) and the
explicit mechanism `project_fine_tuning`, project ID and base model. The commands
below use **total** only as an implementation example; Tanmay's intended meaning
of 5/100 remains unconfirmed.

The selector sorts class names and per-class sample IDs before seeded shuffles.
It uses seed string `<seed>:classes` for class order and
`<seed>:examples:<label>` per class. Total budgets use balanced round-robin;
per-class budgets select that many examples from every class. With fixed inputs,
smaller budgets are prefixes of larger budgets. Insufficient support fails rather
than redistributing examples.

Only the runner's fit-training partition is eligible. Validation and test data
are never uploaded. Repeated IDs and exact UTF-8 text across partitions are
rejected before remote operations.

## Provider flow

For every nonzero condition the adapter:

1. validates the configured project and that the base model supports classification;
2. serializes selected examples to the documented classification JSONL schema;
3. uploads and polls the dataset until upload and profiling finish;
4. creates and polls a classification training job to a terminal status;
5. selects an explicit available checkpoint;
6. creates and polls a deployment, then validates its job, checkpoint, base model,
   task type and exact label set;
7. calls `/v1/classification` with that deployment for held-out examples.

Submission endpoints are not retried because a timeout can leave an accepted
resource whose ID was not returned. `fit_metadata.json` is updated after every
known dataset, job and deployment ID so an interrupted run can be resumed. A
continuation requires the original dataset ID and JSONL SHA-256; job and
deployment continuations additionally require their parent IDs.

## Dry run

This plans 0/5/100 total examples with Banking77 and makes no provider request:

```bash
PYTHONPATH=src python scripts/run_banking77_scaling_benchmark_v2.py \
  --classifiers emissary --class-counts 2 --seeds 42 \
  --train-per-class 70 --test-per-class 1 \
  --min-train-per-class 70 --min-test-per-class 1 --strict-support \
  --validation-fraction .2 --emissary-shots 0 5 100 \
  --emissary-shot-unit total \
  --emissary-mechanism project_fine_tuning \
  --emissary-project-id ms-dry-run-placeholder \
  --emissary-base-model Llama-3.2-1B-Instruct \
  --definitions class_definitions_data/banking77/canonical_llm_enriched_v1.json \
  --dry-run --output-root artifacts/emissary_few_shot
```

The original Banking77 campaign accepts the same Emissary options. It omits the
v2-only minimum-support and `--strict-support` flags.

## Live run safeguards

A live nonzero campaign requires two additional acknowledgements:

- `--emissary-allow-unpriced-training`, because the documented API and live model
  detail provide no price estimate;
- `--emissary-max-training-jobs N`, which must cover the number of new jobs the
  invocation would submit.

For the example above, removing `--dry-run` and adding
`--emissary-allow-unpriced-training --emissary-max-training-jobs 2` permits at
most two new jobs. This bound controls job count, not provider spend. Do not run
it without explicit authorization for the unpriced training calls.

The training hyperparameters and polling limits are configurable through the
`--emissary-num-train-epochs`, `--emissary-learning-rate`, batch-size, timeout,
poll-interval and inactive-timeout options. Defaults match the inspected
`Llama-3.2-1B-Instruct` parameter template where applicable.

## Evidence

Every dry run writes `preparation_plan.json` with selected IDs/text/order,
per-class counts, JSONL byte size and SHA-256, planned operations, mechanism,
comparability statement and unavailable cost. A live run writes
`fit_metadata.json` before inference and after every remote milestone, including
sanitized provider responses and status histories. Signed download URLs are
redacted.

See the [provider contract](emissary_contract.md), [test evidence](emissary_validation.md)
and [small live plan](emissary_smoke_plan.json).
