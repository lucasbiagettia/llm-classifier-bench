# Runner patch

This ZIP is intended to be extracted at the repository root.

It adds only:

```text
src/llm_classifier_bench/runner.py
tests/test_runner.py
```

It does **not** modify the existing Dataset, Classifier, Prediction, Emissary, workflow, or Metric contracts.

## What the runner does

For one `ClassificationDataset × Classifier` run:

```text
load dataset
→ persist run config/sample IDs
→ classifier.fit(train)
→ classifier.predict(test inputs)
→ validate predictions
→ predictions.jsonl
→ metrics.json
```

A run directory contains:

```text
artifacts/runs/<run_id>/
  config.json
  predictions.jsonl
  metrics.json        # when evaluate=True
  status.json
```

`status.json` is updated with the current stage. If a run fails, the failure stage,
exception type, and message are persisted before the exception is re-raised. A future
matrix runner can catch that exception and continue with other jobs.

## Deliberate non-features

The runner does **not** decide how to select 5/10/20 classes, how many examples per
class to sample, whether class subsets are nested, or how many seeds to run. Those are
methodological decisions that are not frozen yet and should not be smuggled into the
runner implementation.

It also does not branch on classifier type. `fit()` is always called; a zero-shot
classifier simply implements it as a no-op.

## Test it

From the repository root:

```bash
PYTHONPATH=src pytest tests/test_runner.py -q
```

The runner tests use fake datasets/classifiers and make no external API calls.

Then run the full unit suite:

```bash
PYTHONPATH=src pytest -m "not integration"
```

## Important before a real API run

Do not point the runner at a paid classifier over a full dataset just to test the
runner. The unit test is enough to validate orchestration. Freeze or build the intended
sampling/subset layer first, then run the actual benchmark configuration.
