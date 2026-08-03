# Classifier Patch Manifest

Extract this ZIP at the repository root with `unzip -o`.

## New files

```text
src/llm_classifier_bench/config.py
src/llm_classifier_bench/classifiers/openai.py
src/llm_classifier_bench/classifiers/bert.py
src/llm_classifier_bench/classifiers/sentence_transformer.py

scripts/_probe_classifier_utils.py
scripts/probe_openai_classifier.py
scripts/probe_bert_classifier.py
scripts/probe_sentence_transformer_classifier.py

tests/classifiers/test_openai.py
tests/classifiers/test_bert.py
tests/classifiers/test_sentence_transformer.py

REGRESSION_TEST_GUIDE.md
PATCH_MANIFEST.md
```

## Modified files

```text
src/llm_classifier_bench/classifiers/base.py
src/llm_classifier_bench/classifiers/emissary.py
src/llm_classifier_bench/classifiers/__init__.py
src/llm_classifier_bench/runner.py
src/llm_classifier_bench/workflows/classification_cycle.py

tests/test_runner.py
tests/workflows/test_classification_cycle.py

requirements.txt
.env.example
README.md
RUNNER_NOTES.md
```

## Intentional contract change

`Classifier` now has the lifecycle:

```text
prepare(classes)
-> fit(train, validation_examples=validation)
-> predict(test)
```

`Prediction` is unchanged.

## Offline validation performed before packaging

```text
32 passed, 3 deselected
```

Command:

```bash
PYTHONPATH=src pytest -m "not integration" -q
```
