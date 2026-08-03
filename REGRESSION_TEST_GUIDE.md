# Regression and Smoke Test Guide

Use this checklist after applying the classifier/runner update. The goal is to verify that the old validated cycle still works before starting measurements.

## 1. Apply the update

From the repository root:

```bash
unzip -o llm-classifier-bench-classifiers.zip
```

Then install/update dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If needed:

```bash
cp .env.example .env
```

Keep your real secrets only in `.env`.

## 2. Run the full offline regression suite

This must pass before making any paid call:

```bash
PYTHONPATH=src pytest -m "not integration"
```

Then run the most important areas independently so failures are easier to localize:

```bash
PYTHONPATH=src pytest tests/classifiers -m "not integration" -q
PYTHONPATH=src pytest tests/datasets -m "not integration" -q
PYTHONPATH=src pytest tests/workflows -q
PYTHONPATH=src pytest tests/metrics -q
PYTHONPATH=src pytest tests/test_runner.py -q
```

Expected result: all existing tests plus the new OpenAI/BERT/SentenceTransformer/runner tests pass.

## 3. Recheck real datasets

These verify that dataset loading still behaves as before:

```bash
PYTHONPATH=src python scripts/probe_dataset.py ag_news
PYTHONPATH=src python scripts/probe_dataset.py banking77
```

Optional real Hugging Face integration tests:

```bash
RUN_HF_INTEGRATION=1 PYTHONPATH=src \
  pytest tests/datasets/test_huggingface_integration.py -m integration -s
```

## 4. Recheck Emissary

First, the API contract probe:

```bash
PYTHONPATH=src python scripts/probe_emissary.py --class-counts 3 5 20
```

Then rerun the already validated AG News semantic smoke using an existing experiment:

```bash
PYTHONPATH=src python scripts/probe_ag_news_emissary.py \
  --examples-per-class 12 \
  --model-id <experiment_id>/<version>
```

The exact accuracy does not have to equal the historical smoke result. What matters here is that the API call succeeds, normalized probabilities remain valid, the artifact is written, and evaluation runs.

Re-evaluate the artifact explicitly:

```bash
PYTHONPATH=src python scripts/evaluate_artifact.py \
  artifacts/probes/<new-artifact>.jsonl
```

## 5. Smoke-test OpenAI

Put `OPENAI_API_KEY` in `.env`, then run:

```bash
PYTHONPATH=src python scripts/probe_openai_classifier.py
```

The default runs one AG News test example per class, so it makes only a handful of paid requests.

Verify:

- the run finishes;
- `predictions.jsonl` has one row per requested sample;
- every predicted label belongs to AG News;
- `confidence` is `null`;
- `probabilities` is empty in the serialized artifact;
- accuracy/macro-F1/latency are available;
- probability-dependent metrics are unavailable rather than fabricated.

## 6. Smoke-test SentenceTransformer + Logistic Regression

Run:

```bash
PYTHONPATH=src python scripts/probe_sentence_transformer_classifier.py
```

First execution downloads the sentence-transformer model.

Verify:

- train/validation IDs are disjoint in `config.json`;
- test IDs appear in neither fit-train nor validation;
- the classifier produces a complete probability distribution over all AG News classes;
- probabilities sum approximately to 1;
- calibration/log-loss/Brier metrics are available.

## 7. Smoke-test BERT fine-tuning

Run:

```bash
PYTHONPATH=src python scripts/probe_bert_classifier.py
```

The default smoke configuration uses a tiny AG News sample and one epoch. It is intentionally a functional check, not a performance estimate.

Verify the same split invariants as above and confirm that BERT produces complete class probabilities.

If CPU training is inconvenient, the smoke size can be reduced further, but do not interpret its accuracy scientifically.

## 8. Inspect runner artifacts

For each of the three new classifier runs, inspect:

```bash
cat artifacts/runs/<run_id>/status.json
cat artifacts/runs/<run_id>/config.json
head artifacts/runs/<run_id>/predictions.jsonl
cat artifacts/runs/<run_id>/metrics.json
```

The key scientific invariant is:

```text
fit_train IDs ∩ validation IDs = empty
fit_train IDs ∩ test IDs       = empty
validation IDs ∩ test IDs      = empty
```

The final `test` IDs must come directly from `DatasetBundle.test` and must never be passed into classifier training/model selection.

## 9. Regression gate before measurements

Start pilot measurements only if all of the following are true:

- offline tests pass;
- AG News and Banking77 still load;
- Emissary real smoke succeeds;
- OpenAI real smoke succeeds;
- SentenceTransformer real smoke succeeds;
- BERT real smoke succeeds;
- runner artifacts show correct train/validation/test separation;
- old JSONL artifacts can still be evaluated with `evaluate_artifact.py`.

At that point the software cycle is closed enough for pilot benchmark execution.

## 10. What is still not a formal benchmark decision

Passing this guide proves infrastructure compatibility, not scientific validity. Before publishable runs, freeze class-description generation, class subsets, sample sizes, seeds/repetitions, and supervision-budget reporting.
