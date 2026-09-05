# TF-IDF issue #8 validation

Verified on 2026-09-05 using the existing repository `venv` (Python 3.14.7,
scikit-learn 1.9.0, NumPy 2.5.1, SciPy 1.18.0). No dependencies were upgraded.

- `PYTHONPATH=src venv/bin/pytest -m "not integration"`: **65 passed, 3 deselected**.
  No pre-existing offline failures. The three external integration tests were
  intentionally excluded.
- `PYTHONPATH=src venv/bin/python scripts/probe_tfidf_classifier.py --run-id issue8-local-fixture`:
  completed, including unseen-token inference, predictions, metrics, and fit metadata.
  Local output: `artifacts/tfidf_smoke/issue8-local-fixture/`.
- Both campaign entry points ran end to end on local fixtures in tests, with API
  and transformer classifier construction forbidden. CSV/JSON summaries and
  replay from saved settings/sample IDs passed.
- Real Banking77 v2 campaign from the existing local Arrow cache: **1 completed,
  0 failed**, 5 classes, seed 42, 12 source-train and 2 test examples per class,
  25% validation, strict support. Actual partitions: **45 fit, 15 validation,
  10 held-out test**. Selected C: **10.0**. Local output:
  `artifacts/tfidf_validation_cached/20260905T173829Z/`.
- Reconstructed that real-data classifier from its saved training configuration
  and exact sample IDs: selected C and all 10 probability mappings matched exactly.
- `git diff --check`: passed.

These are execution checks, not publishable quality estimates. No paid API calls,
model downloads, transformer training, or full benchmark runs were performed.
Existing benchmark artifacts and unrelated working-tree changes were preserved.
New validation artifacts remain local and are not part of the source commit.

## Cached Banking77 loader limitation and executed command

The ordinary campaign CLI was attempted with `HF_HUB_OFFLINE=1` and
`HF_DATASETS_OFFLINE=1`, but the existing Hugging Face CSV loader still tried to
resolve remote data-file URLs through fsspec. That attempt was interrupted before
any classifier ran. To run entirely offline, the successful validation used the
existing dataset adapter's `loader` injection point with `Dataset.from_file`.
The campaign implementation, source train/test rows, label selection, sampling,
train/validation splitting, and evaluation were unchanged.

The existing `csv` cache was copied to `/tmp/issue8-hf-cache/` to preserve the
original cache. The temporary `/tmp/issue8_cached_campaign.py` harness contained:

```python
from pathlib import Path
from datasets import Dataset
from llm_classifier_bench.datasets.huggingface import HuggingFaceClassificationDataset
from llm_classifier_bench.datasets.registry import BANKING77_SPEC
import run_banking77_scaling_benchmark_v2 as campaign

cache = Path('/tmp/issue8-hf-cache/csv/default-a38433035ea47098/0.0.0/d41f37fffd4cc4dfd07485b661c45b9863c2d0a8b0a28faa84befecfef33631a')

def loader(path, *, split, **kwargs):
    return Dataset.from_file(str(cache / f'csv-{split}.arrow'))

campaign.get_dataset = lambda name: HuggingFaceClassificationDataset(BANKING77_SPEC, loader=loader)
campaign.main()
```

Executed:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src:scripts \
  venv/bin/python /tmp/issue8_cached_campaign.py \
  --classifiers tfidf --class-counts 5 --seeds 42 \
  --train-per-class 12 --test-per-class 2 \
  --min-train-per-class 12 --min-test-per-class 2 --strict-support \
  --validation-fraction 0.25 --output-root artifacts/tfidf_validation_cached
```

The normal campaign command and focused saved-configuration replay snippet are in
[README.md](README.md#tf-idf--logistic-regression). The standalone local fixture
probe is the supported smoke command that needs neither network nor a dataset cache.
