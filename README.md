# LLM Classifier Benchmark

Reproducible benchmark infrastructure for comparing closed-set text-classification approaches across predictive quality, probabilistic calibration, latency, and cost.

The current implementation supports five classifier families:

- **Emissary zero-shot and Projects fine-tuning** — routing experiments plus an
  explicit supervised classification mechanism.
- **OpenAI zero-shot** — generative closed-set classification with structured output.
- **BERT fine-tuned** — supervised Hugging Face sequence classification.
- **Frozen SentenceTransformer + Logistic Regression** — supervised shallow classifier over fixed semantic embeddings.
- **TF-IDF + Logistic Regression** — supervised sparse lexical baseline.

## Documentation

- [Experimental protocol and interpreting results](docs/experimental_protocol_v2.md)
- [Frozen class definitions](docs/class_definitions.md)
- [Matched labeled-example budgets](docs/matched_label_budgets.md)
- [Emissary configuration](docs/emissary_few_shot.md) and [adapter contract](docs/emissary_contract.md)
- [Latency and throughput](docs/latency_measurement.md)
- [Inference costs](docs/inference_costs.md), [self-hosted projections](docs/self_hosted_inference_costs.md), and [preparation costs](docs/preparation_costs.md)

Generated runs and reports belong in `artifacts/`, which is ignored by Git.
The repository ships code, frozen class definitions and example rate cards;
it does not include pilot results or claim a published benchmark ranking.
Keep complete run directories when sharing results so measurements can be audited.

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
OpenAI:             gpt-5-nano
OpenAI reasoning:   minimal
BERT:               google-bert/bert-base-uncased
SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
```

The OpenAI default is a model alias. For a reproducible campaign, select an
available dated model identifier explicitly with `--openai-model` and retain the
resolved model recorded in the artifacts.

## Supervision regimes

The classifiers are not all methodologically equivalent:

```text
Emissary routing     zero-shot
Emissary Projects    supervised fine-tuning
OpenAI               zero-shot
BERT                  supervised fine-tuning
SentenceTransformer   supervised embeddings + Logistic Regression
TF-IDF                supervised lexical features + Logistic Regression
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

OpenAI uses strict JSON Schema Structured Outputs and explicitly sets GPT-5 reasoning effort to `minimal` for the latency-sensitive classification baseline. The requested model identifier and reasoning effort are persisted in run configuration.

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

Accuracy ties retain the first configured C. The runner saves candidate scores,
selected C, and probability label order in `fit_metadata.json`.
Validation-accuracy ties can select a strongly regularized, underconfident model;
accuracy and probability quality should be reported separately.

This is intentionally different from BERT fine-tuning: it measures the strength of frozen semantic features plus a shallow supervised classifier.

### TF-IDF + Logistic Regression

`TfidfLogisticClassifier` uses sparse word unigrams/bigrams, lowercase text, no
stop-word list, smoothed IDF, and L2 normalization. `TfidfTrainingConfig` controls
ngram range, lowercase, minimum document frequency, maximum vocabulary size,
sublinear TF, C candidates, fallback C, maximum iterations, and seed.

Vocabulary, IDF, and all candidate weights are fitted exclusively on fit-train.
Validation accuracy selects C from `(0.1, 1.0, 10.0)`; ties retain the **first
candidate in configured order**. Without validation, only fixed `fallback_c=1.0`
is fitted. There is no train-plus-validation refit, training-accuracy selection,
or classifier-side sampling. Every prepared class must appear in fit-train.
Unknown tokens can yield a zero vector and still receive ordinary probabilities.
Latency includes per-example text transformation and probability inference.

Both maintained campaign entry points, `run_banking77_scaling_benchmark.py` and
`run_banking77_scaling_benchmark_v2.py`, support `--classifiers tfidf`; existing
default classifier lists are unchanged. They use the same condition sampling and
runner split as other classifiers, and include TF-IDF in summary CSV/JSON quality,
calibration, and latency results.
TF-IDF-only selection creates no API clients, generates no definitions, and loads
no transformer models. The campaigns reuse the checked-in definition profile.

Run a small CPU-only local fixture without credentials or downloads:

```bash
PYTHONPATH=src python scripts/probe_tfidf_classifier.py
```

Run a small TF-IDF-only Banking77 campaign (the CSV loader may require network access):

```bash
PYTHONPATH=src python scripts/run_banking77_scaling_benchmark_v2.py \
  --classifiers tfidf --class-counts 5 --seeds 42 \
  --train-per-class 12 --test-per-class 2 \
  --min-train-per-class 12 --min-test-per-class 2 --strict-support \
  --validation-fraction 0.25 --output-root artifacts/tfidf_campaign
```

For the original campaign entry point, use the same command with its filename and
omit the three v2 support options (`--min-train-per-class`, `--min-test-per-class`,
`--strict-support`). Both expose `--tfidf-c-values`, `--tfidf-fallback-c`,
`--tfidf-ngram-range`, `--[no-]tfidf-lowercase`, `--tfidf-min-df`,
`--tfidf-max-features`, `--tfidf-sublinear-tf`, and `--tfidf-max-iter`.
The Banking77 CSV loader may resolve source URLs even when a dataset cache exists;
the local fixture above is the supported smoke check without network access.
Smoke results verify execution and are not publishable benchmark findings.

#### Saved settings and reproduction

`config.json` preserves training settings, exact fit/validation/test sample IDs,
class definitions, split seed, and dataset identity. The generic optional
`fitted_metadata()` hook writes `fit_metadata.json` immediately after fitting,
without changing the pre-fit configuration. For TF-IDF this records actual
supervision counts, vectorizer settings/vocabulary size, C candidates and scores,
selected/fallback C, selection rule, solver, seed, and scikit-learn/NumPy/SciPy
versions. `campaign.json` also records the TF-IDF settings per seed and shared
sampling budgets. Predictions and probability metrics use the existing artifacts.

To reproduce a Banking77 run, retain the same source dataset and definition
profile, use the recorded library versions, and rerun the campaign with its saved
budgets, seeds, and TF-IDF options into a **new** output directory. Check sample IDs
against the original run; source data changes can invalidate seed-only reproduction.
Alternatively, the following focused snippet reconstructs the fitted classifier
from a run's saved settings and sample selection and checks its probabilities
(latency is expected to differ):

```python
import json
from pathlib import Path
from llm_classifier_bench.classifiers import TfidfLogisticClassifier
from llm_classifier_bench.config import TfidfTrainingConfig
from llm_classifier_bench.core import ClassDefinition
from llm_classifier_bench.datasets import get_dataset

run = Path("artifacts/tfidf_campaign/<campaign>/runs/<run>")
config = json.loads((run / "config.json").read_text())
fit = json.loads((run / "fit_metadata.json").read_text())
settings = config["classifier"]["training"]
for key in ("ngram_range", "c_values"):
    settings[key] = tuple(settings[key])
bundle = get_dataset("banking77").load()
by_id = {e.sample_id: e for e in bundle.train + bundle.test}
data = config["dataset"]
classifier = TfidfLogisticClassifier(training=TfidfTrainingConfig(**settings))
classifier.prepare([ClassDefinition(**c) for c in data["classes"]])
classifier.fit([by_id[i] for i in data["fit_train_sample_ids"]],
               validation_examples=[by_id[i] for i in data["validation_sample_ids"]])
predictions = classifier.predict([by_id[i].as_input() for i in data["test_sample_ids"]])
saved = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
assert classifier.selected_c == fit["selected_c"]
assert [p.probabilities for p in predictions] == [p["probabilities"] for p in saved]
```

For the local smoke run use `fixture_bundle()` from
`scripts/probe_tfidf_classifier.py` in place of `get_dataset("banking77").load()`.
Both campaign replay paths are covered by offline tests with real scikit-learn.

## Runner

`src/llm_classifier_bench/runner.py` is classifier-agnostic.

It performs:

```text
load dataset
-> split train/validation
-> persist config/sample IDs
-> prepare classifier
-> fit classifier
-> persist optional fit_metadata.json
-> predict untouched test in batches
-> validate and append each successful batch to predictions.jsonl
-> atomically reconcile prediction costs after inference
-> compute metrics.json
```

The runner never branches on classifier names.

## Latency and throughput

The runner now times complete `predict()` calls, including preprocessing and
output normalization, and saves individual observations in `timings.jsonl`.
`operational_report.json` separates preparation, warmup, successful calls and
failed calls; it reports P50, P95, observation counts and throughput. P99 requires
at least 1,000 observations. Batched-call amortized time is reported separately
from individual-request latency.

Both campaign scripts accept `--inference-batch-size` (default 1),
`--warmup-examples` (default 0), `--client-location` and `--cache-condition`.
Warmup makes extra predictions on fit-training examples; hosted calls may be
billed. Without explicit warmup the run is labeled unwarmed. The default OpenAI
client uses no automatic retries and a 120-second timeout.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/probe_latency.py --run-id local-timing

PYTHONPATH=src python scripts/report_operational.py \
  artifacts/latency_smoke/local-timing --output /tmp/local-timing.md
```

See the [measurement contract and validation](docs/latency_measurement.md) for
timing boundaries, metadata, failure handling, and historical-artifact compatibility.

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

`config.json` records the exact fit-train, validation, and final-test sample IDs plus split seed and classifier configuration where available. For zero-shot classifiers it also records `training_examples_used=0` and `validation_examples_used=0`, so the existence of dataset train partitions cannot be mistaken for labeled supervision actually consumed by the classifier.

`predictions.jsonl` is the primary reproducibility artifact. Each validated batch is
saved before the next inference call. A later failure or Ctrl-C preserves earlier
batches, and Ctrl-C is recorded as a failed call/run before being re-raised.
Incomplete rows have `cost_reconciled=false` and unavailable per-prediction costs;
the usage ledger remains the accounting source. Final cost enrichment replaces
the file atomically, so a reporting failure cannot truncate saved predictions.
This does not add automatic resume or recover results inside a failed batch.

Metrics can be recalculated without repeating model/API inference. Duplicate
sample IDs are rejected. For runner artifacts with a usage ledger, reevaluation
requires a completed measurement and the full test population; retained partial
predictions are not treated as a complete benchmark.

Adaptive ECE keeps identical confidence values in one bin. Equal-frequency
boundaries move to the end of a tie, possibly reducing the number of bins. Its
metadata identifies this policy as `equal_frequency_preserve_ties_v2`. Historical
metrics are not rewritten automatically; reevaluation may change Adaptive ECE
when ties crossed old bin boundaries.

The Emissary adapter, runner and default probability metrics share an absolute
sum tolerance of `1e-4` (zero relative tolerance). Small rounding errors are
accepted consistently without changing the reported probabilities; non-finite,
out-of-range values and larger sum errors remain invalid.

Classifier inputs retain only provenance metadata (`dataset`, `source`, `split`,
`row_index`). Raw labels and arbitrary target-bearing dataset metadata stay on
the labeled examples and are not forwarded by `as_input()`.

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

## Emissary labeled-example budgets

Both maintained Banking77 campaign scripts accept `--emissary-shots 0 5 100`,
`--emissary-shot-unit total|per_class`, `--emissary-selection-seed`, and
`--emissary-selection-policy balanced_round_robin_v1`. Zero-shot remains the
default routing-experiment path. Nonzero budgets can use the documented Projects
classification fine-tuning flow with `--emissary-mechanism project_fine_tuning`,
an existing project ID and an explicit base model.

Projects fine-tuning trains a separately selected base model; it does not retrain
the routing experiment. Report that mechanism/model confound when comparing it
with zero-shot. Live training additionally requires an explicit unknown-cost
acknowledgement and a bound on newly created training jobs.

See [configuration, commands and artifacts](docs/emissary_few_shot.md) and
[the adapter contract and limitations](docs/emissary_contract.md).

## Inference cost accounting

Runs save `usage.jsonl`, `pricing.json` and `cost_report.json`, including failed
attempts and warmup. Costs are observed, estimated, or unavailable; missing prices
never mean free inference. Both maintained campaigns accept an optional
`--pricing pricing/inference_2026-09-16.json` rate card. Reprice without new calls
using `scripts/report_costs.py`.

See [inference cost conventions and commands](docs/inference_costs.md) and
[preparation investment and amortization](docs/preparation_costs.md).

For local classifiers, cost reports also show continuous-throughput projections
and optional deployed-allocation costs including idle time. Use
`report_costs.py --deployment-hours 24 --volumes 1000 10000 100000`; this only
replays saved evidence. See [self-hosted inference cost scenarios](docs/self_hosted_inference_costs.md)
for hardware/rate matching, input-length metadata and feasibility checks.


## Preparation investment and amortization

Runs also save `preparation.jsonl`, `preparation_metadata.json` and
`preparation_report.json`. These measure loading/features, candidate fitting and
selection without counting nested work twice. The winning TF-IDF/MiniLM fit is a
component of the total; no final refit is introduced. Remote waiting time is not
assumed to be billable compute.

Use `scripts/report_preparation.py RUN_DIR --inference-scenario continuous` or
`--inference-scenario deployed --deployment-hours 24` to combine preparation with
saved inference scenarios for 1,000 / 10,000 / 100,000 valid predictions. Hosted
APIs use `--inference-scenario recorded_api`. Unknown components remain unavailable.
See [accounting rules, evidence formats, commands and validation](docs/preparation_costs.md).

## Matched labeled-example budgets

Both Banking77 campaign scripts accept `--matched-budgets 0 5 100 --budget-unit total`
(or explicit `per_class`) and a separate `--validation-budget`, default 0. They
save shared fit/context IDs, class coverage, actual label consumption and explicit
unsupported cells. OpenAI uses the shared pool as demonstrations and requires an
explicit context-window setting. Full-training references remain a separate mode.
See [supported matrix and commands](docs/matched_label_budgets.md).
