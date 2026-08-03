# LLM Classifier Benchmark

Reproducible benchmark infrastructure for comparing closed-set text-classification approaches across predictive quality, probabilistic calibration, latency, and cost.

The current implementation supports four classifier families:

- **Emissary zero-shot** — purpose-built discriminative classification API.
- **OpenAI zero-shot** — generative closed-set classification with structured output.
- **BERT fine-tuned** — supervised Hugging Face sequence classification.
- **Frozen SentenceTransformer + Logistic Regression** — supervised shallow classifier over fixed semantic embeddings.

## Experimental lifecycle

Every classifier implements the same lifecycle:

```text
prepare(classes)
    -> fit(train, validation_examples=validation)
    -> predict(test)
    -> Prediction[]
    -> JSONL artifact
    -> metrics.json
```

The dataset's original `test` split is always treated as the final held-out benchmark set. It is never passed to `prepare()` or `fit()`.

Only the original training split is partitioned:

```text
DatasetBundle.train
    -> deterministic stratified split
       -> fit_train
       -> validation

DatasetBundle.test
    -> untouched final benchmark holdout
```

Zero-shot classifiers ignore the fit/validation examples. Supervised classifiers use them for training and model selection.

## Default models

Defaults live in:

```text
src/llm_classifier_bench/config.py
```

Current defaults:

```text
OpenAI:             gpt-5-nano-2025-08-07
BERT:               google-bert/bert-base-uncased
SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
```

The OpenAI snapshot is intentionally pinned for reproducibility. Change the config deliberately when benchmarking another model.

## Supervision regimes

The classifiers are not all methodologically equivalent:

```text
Emissary             zero-shot
OpenAI               zero-shot
BERT                  supervised fine-tuning
SentenceTransformer   supervised embeddings + Logistic Regression
```

Results should therefore report the supervision regime and number of labeled training examples. Do not interpret a supervised model winning as evidence that the underlying technique is universally superior to a zero-shot method.

## Classifier contract

`src/llm_classifier_bench/classifiers/base.py` defines:

```python
class Classifier(Protocol):
    @property
    def name(self) -> str: ...

    def prepare(self, classes: Sequence[ClassDefinition]) -> None: ...

    def fit(
        self,
        examples: Sequence[LabeledExample],
        *,
        validation_examples: Sequence[LabeledExample] = (),
    ) -> None: ...

    def predict(
        self,
        examples: Sequence[ClassificationInput],
    ) -> list[Prediction]: ...
```

`Prediction` remains the common output contract. Probabilities and confidence are optional.

OpenAI intentionally returns `confidence=None` and `probabilities=None`; the benchmark does not use model self-reported confidence as probabilistic evidence. Metrics that require probabilities therefore become unavailable for that classifier.

## Supervised classifiers

### BERT

`BertClassifier` fine-tunes a Hugging Face `AutoModelForSequenceClassification` end-to-end.

Validation examples are used to select the best epoch by validation loss. The best epoch state is restored before final test inference.

### SentenceTransformer + Logistic Regression

`SentenceTransformerLogisticClassifier` keeps the sentence-transformer encoder frozen.

It:

1. embeds training examples;
2. trains Logistic Regression candidates over a configured `C` grid;
3. selects `C` using validation accuracy;
4. predicts probabilities on the final held-out test set.

This is intentionally different from BERT fine-tuning: it measures the strength of frozen semantic features plus a shallow supervised classifier.

## Runner

`src/llm_classifier_bench/runner.py` is classifier-agnostic.

It performs:

```text
load dataset
-> split train/validation
-> persist config/sample IDs
-> prepare classifier
-> fit classifier
-> predict untouched test
-> validate Prediction contract
-> write predictions.jsonl
-> compute metrics.json
```

The runner never branches on classifier names.

## Installation

Target environment:

```text
Python 3.12
```

Create the environment and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set the credentials you use:

```dotenv
EMISSARY_API_KEY=...
OPENAI_API_KEY=...
HF_TOKEN=...
```

## Unit tests

Unit tests never call paid APIs or download models:

```bash
PYTHONPATH=src pytest -m "not integration"
```

Classifier-only tests:

```bash
PYTHONPATH=src pytest tests/classifiers -m "not integration" -q
```

Runner tests:

```bash
PYTHONPATH=src pytest tests/test_runner.py -q
```

## Real smoke tests

These are infrastructure validation runs, not publishable benchmark results.

### OpenAI

Makes a small number of paid API calls:

```bash
PYTHONPATH=src python scripts/probe_openai_classifier.py
```

Override the model:

```bash
PYTHONPATH=src python scripts/probe_openai_classifier.py \
  --model gpt-5-nano-2025-08-07
```

### SentenceTransformer + Logistic Regression

Downloads the embedding model on first use:

```bash
PYTHONPATH=src python scripts/probe_sentence_transformer_classifier.py
```

### BERT fine-tuning

Downloads the model and performs a deliberately tiny one-epoch fine-tune:

```bash
PYTHONPATH=src python scripts/probe_bert_classifier.py
```

This smoke run is intentionally small and says nothing meaningful about BERT benchmark quality.

### Existing Emissary checks

Contract probe:

```bash
PYTHONPATH=src python scripts/probe_emissary.py --class-counts 3 5 20
```

Existing AG News smoke:

```bash
PYTHONPATH=src python scripts/probe_ag_news_emissary.py \
  --examples-per-class 12 \
  --model-id <experiment_id>/<version>
```

## Artifacts

Each runner execution writes:

```text
artifacts/runs/<run_id>/
  config.json
  predictions.jsonl
  metrics.json
  status.json
```

`config.json` records the exact fit-train, validation, and final-test sample IDs plus split seed and classifier configuration where available.

`predictions.jsonl` is the primary reproducibility artifact. Metrics can be recalculated from it without repeating model/API inference.

## Datasets

Currently supported:

- `ag_news`
- `banking77`

AG News is suitable for smoke validation. Banking77 is the intended multiclass benchmark source for larger label spaces.

## Before formal benchmark claims

The infrastructure is sufficient to begin pilot measurements after the regression/smoke suite passes. Before treating results as publishable benchmark evidence, still freeze:

1. canonical/reproducible class descriptions;
2. 5/10/20-class subset-selection policy;
3. number of train/test examples per condition;
4. seeds/repetitions and confidence-interval procedure;
5. supervision-regime reporting and training-budget policy.

Do not use the existing small smoke runs as scientific evidence.


## Contributing

### Setup

Follow the Installation steps above, then install dev dependencies if listed separately in `requirements.txt` or a `requirements-dev.txt`.

### Before opening a PR

1. Run the unit tests:

```bash
   PYTHONPATH=src pytest -m "not integration"
```

2. If you touched a specific classifier or the runner, also run its targeted tests:

```bash
   PYTHONPATH=src pytest tests/classifiers -m "not integration" -q
   PYTHONPATH=src pytest tests/test_runner.py -q
```

3. Format and lint (adjust to whatever tools the project uses, e.g. `black`, `ruff`, `mypy`).

Integration tests that call paid APIs or download models are not run in CI by default; only run them locally when relevant, and never rely on them as a merge gate.

### Guidelines

- New classifiers must implement the `Classifier` protocol in `src/llm_classifier_bench/classifiers/base.py` and respect the existing lifecycle (`prepare -> fit -> predict`).
- The runner must remain classifier-agnostic — don't add classifier-specific branches to `runner.py`.
- Never let `fit()` or `prepare()` see the test split; the held-out test set must stay untouched.
- Report the supervision regime (zero-shot vs. supervised) for any new classifier, and don't frame results as benchmark-ready without following the "Before formal benchmark claims" checklist.
- Keep commits focused and add/update tests for any behavior change.

### Pull requests

- Branch from `main`, use a descriptive branch name (e.g. `feat/add-xyz-classifier`).
- Describe what changed and why; link related issues if any.
- Keep PRs scoped to one logical change when possible.