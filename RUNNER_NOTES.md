# Runner notes

The runner now supports both zero-shot and supervised classifiers through one lifecycle:

```text
load dataset
-> split DatasetBundle.train into fit_train + validation
-> persist exact split/sample IDs
-> classifier.prepare(classes)
-> classifier.fit(fit_train, validation_examples=validation)
-> classifier.predict(DatasetBundle.test)
-> validate normalized predictions
-> predictions.jsonl
-> metrics.json
```

The original dataset test split is the final holdout and is never used for training or model selection.

Zero-shot classifiers implement `fit()` as a no-op. The runner never branches on classifier type.

The runner still does not decide the formal 5/10/20-class subset policy or benchmark repetitions. Those remain explicit experimental-protocol decisions.

Run its unit tests with:

```bash
PYTHONPATH=src pytest tests/test_runner.py -q
```
