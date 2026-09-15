# Calibration audit — issue #7

Audited 2026-09-15. **The five-class MiniLM + LR result is valid and explained
by validation-accuracy tie breaking. No historical metric values need correction.**
The audit also closes a separate input-validation gap when evaluating saved files.

## Evidence and original values

Campaign: [`20260804T014841Z`](../artifacts/benchmark_runs/20260804T014841Z/campaign.json).
All 16 completed runs have `config.json`, `predictions.jsonl`, and `metrics.json`.
The two main runs are
`20260804T014841Z__seed42__n05__sentence-transformer-logreg` and
`20260804T014841Z__seed42__n10__sentence-transformer-logreg`, under that campaign's
`runs/` directory. Each prediction saves `raw_response.selected_c`.
Original candidate validation scores and fitted LR weights were not saved.

Both use Banking77, frozen `sentence-transformers/all-MiniLM-L6-v2`, seed 42,
100 sampled source-training examples per class, and 20 held-out test examples per
class. The 20% validation split leaves **80 fit + 20 validation examples per class**:
400/100/100 fit/validation/test rows for K=5; 800/200/200 for K=10.
LR uses `C=(0.1, 1, 10)`, `max_iter=2000`, and validation accuracy for selection.
No training examples enter an inference prompt.

| Quantity | 5 classes | 10 classes |
| --- | ---: | ---: |
| Selected C | 0.1 | 10 |
| Accuracy | 0.970000 | 0.970000 |
| Mean confidence | 0.550502 | 0.922571 |
| ECE, 10 fixed-width bins | 0.419497580 | 0.048742768 |
| Adaptive ECE, 10 equal-frequency bins | 0.419497580 | 0.047428636 |
| Log loss | 0.621821718 | 0.121769367 |
| Multiclass Brier score | 0.274136334 | 0.047643042 |

**Original and recalculated values are identical**, including for K=20 and K=25
and the other classifiers. Independent NumPy implementations agree within 1e-12.
The [machine-readable report](calibration_audit_v2_results.json) contains every
original/recalculated/reference value, checks, source SHA-256 hashes, and replay
results. Originals were preserved.

### Label mapping, normalization, and metric conventions

- MiniLM pairs `predict_proba` columns with the fitted LR's `classes_`, rather than
  configured class order. Replay confirms the label mapping; all predicted labels
  match the historical artifacts. A regression test explicitly reverses the
  configured versus learned order.
- Every saved distribution has exactly the configured labels, finite values in
  [0, 1], a sum within tolerance, a predicted label at the maximum probability,
  and matching confidence. Largest sum error: 9.69e-8 for K=5, 2.21e-7 for K=10,
  and 3.01e-7 across the campaign.
- ECE weights each nonempty bin's absolute accuracy–confidence gap by sample
  count. Fixed bins use `min(int(confidence * 10), 9)`, including confidence 1.
  Adaptive bins sort stably by confidence and split into sizes differing by at
  most one; ties may span bins and therefore depend on saved row order.
- Log loss uses natural logarithms of gold-label probabilities, clipped to
  [1e-15, 1−1e-15]. No renormalization occurs. Brier averages the **sum** of squared
  classwise errors over samples (range [0, 2]), without dividing by class count.
- The four label-only OpenAI runs retain unavailable ECE, adaptive ECE, log loss,
  and Brier. Confidence alone permits ECE; log loss and Brier require the full
  distribution. Historical `{}` and `null` probability fields both mean missing
  outputs in the JSONL adapter. No probabilities are synthesized.
- Test IDs and their order exactly match saved configurations; partitions have
  unique IDs and no fit/validation/test ID overlap within a run.

## Why five classes look underconfident

Offline reconstruction uses the **exact saved fit/validation/test IDs and order**,
the cached Banking77 Arrow files, and cached MiniLM weights. Cached test text and
gold labels match the saved predictions. Only LR is fitted; the encoder stays
frozen. Candidate results below are a diagnostic replay, not replacement scores.

| K | C | Validation accuracy | Validation log loss | Test mean confidence | Test ECE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | **0.1** | **0.970** | 0.612350 | 0.550502 | 0.419498 |
| 5 | 1 | 0.970 | 0.164932 | 0.867332 | 0.118887 |
| 5 | 10 | 0.970 | 0.061671 | 0.964144 | 0.033711 |
| 10 | 0.1 | 0.950 | 1.003108 | 0.377264 | 0.517736 |
| 10 | 1 | 0.955 | 0.281550 | 0.771571 | 0.178429 |
| 10 | **10** | **0.975** | 0.107007 | 0.922571 | 0.048743 |

Selection updates only when `score > best_score`. At K=5 all three candidates
tie, so the first candidate, C=0.1, wins. At K=10, C=10 has strictly better
validation accuracy. Smaller C means stronger regularization
([scikit-learn reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)).
Within the five-class replay, increasing C from 0.1 to 10 increases coefficient
L2 norm from 5.125 to 20.316 and raises mean confidence from 55.1% to 96.4%.
This controlled comparison supports regularization as the explanation for the
underconfidence. All populated K=5 confidence bins are underconfident, making
both ECE variants equal to accuracy minus mean confidence: 0.97−0.55050242.

The replay recovers both selected C values and **all historical predicted labels**.
Maximum absolute probability differences from the originals are 3.58e-7 (K=5)
and 2.03e-5 (K=10). It is numerically close, not bitwise identical. Original model
revision/library versions were not pinned in the run configurations, so an exact
historical environment cannot be established. Replay records the cached model
snapshot, file hashes, and current library versions in the JSON report.

### Additional limitation relevant to issue #6

The source-training pools and test examples are nested between K=5 and K=10,
but **fit/validation membership is not held fixed for shared classes**.
`split_train_validation` uses one RNG across alphabetically sorted classes;
adding labels changes its state before shuffling existing labels. Only 316/400
fit IDs and 16/100 validation IDs keep their partition; 84 IDs move each way.
There is no within-run leakage, but cross-K differences cannot be attributed
solely to adding candidate labels. The within-K C ablation above avoids this
confound. Freeze per-label fit/validation membership before any final evaluation
that claims to isolate label-count effects; record this dependency in issue #6's
protocol. This audit does not change historical splits or model selection.

## Fixes, validation, and repeat policy

- The runner already validates prediction/confidence consistency, but direct
  artifact evaluation bypassed those checks. ECE could trust an explicit
  confidence inconsistent with probabilities or derive confidence from a
  malformed distribution. Calibration metrics now reject malformed distributions,
  missing predicted labels, non-argmax predictions, and inconsistent confidence.
  The JSONL adapter also rejects unlabeled probability arrays with file/line
  context. **None of the 16 audited runs has these defects.**
- MiniLM now exposes the existing runner `fitted_metadata` hook, saving candidate
  validation accuracies, selected C, tie rule, probability label order, fit and
  validation counts, and numeric-library versions in `fit_metadata.json` for
  future runs. Selection behavior remains the same.
- Regression checks cover column alignment, first-candidate ties, inconsistent
  metric inputs, probability-map order, bin boundaries, and unavailable outputs.
  `PYTHONPATH=src venv/bin/pytest -m 'not integration'`: **145 passed,
  3 deselected**. No paid API calls or downloads were made.
- **Runs requiring repetition because artifacts are missing or irreparable:
  none in this campaign.** No corrected historical values are needed. Changing
  C selection (for example using validation log loss to break accuracy ties),
  calibration, or split policy constitutes a new experiment and must be declared
  before final evaluation. The diagnostic C=10 five-class result must not replace
  the original result after inspecting test performance.

## Reproduce

Recalculate all saved metrics and independent checks without loading a model:

```bash
PYTHONPATH=src venv/bin/python scripts/audit_calibration.py \
  --output /tmp/calibration_audit.json
```

To also replay the C grid, provide existing local files (no remote loader):

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src \
venv/bin/python scripts/audit_calibration.py \
  --output /tmp/calibration_audit_with_replay.json \
  --train-arrow /path/to/csv-train.arrow \
  --test-arrow /path/to/csv-test.arrow \
  --model-path /path/to/local/MiniLM/snapshot
```

Executed with the local `csv/default-a38433035ea47098/0.0.0/` dataset cache
revision `d41f37fffd4cc4dfd07485b661c45b9863c2d0a8b0a28faa84befecfef33631a`
and MiniLM snapshot `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
Full source hashes and library versions are in the report.
