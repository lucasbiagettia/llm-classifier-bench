# Issue #13 validation evidence

Validated 2026-09-06 America/Mexico_City / 2026-09-07 UTC with the existing
Python 3.14.7 environment; no package upgrades.

- `PYTHONPATH=src venv/bin/pytest -m "not integration"`: **93 passed, 3 deselected**.
- Focused new tests: **28 passed**. Selection determinism/input-order independence,
  nested budgets, total/per-class counts, partial coverage, support errors,
  validation/test ID and exact-content leakage, invalid configuration, rejection
  of every nonzero live budget and reference-model reuse, no eager client creation,
  unchanged test IDs/definitions in both campaigns, persisted plans/configuration,
  real-runner mocked HTTP serialization, pinned version/class-space validation,
  zero-shot metadata and unknown cost, ambiguous creation timeout without retry,
  state reset and post-fit evidence surviving inference failure.
- Cached real Banking77 v2 dry run: **3 planned, 0 failed**, budgets **0/5/100
  total**, **112 fit / 28 validation / 2 test**, seed 42, two classes. Classes:
  `beneficiary_not_allowed`, `wrong_amount_of_cash_received`. Artifact checks
  confirmed identical class definitions and test IDs, and the 5-example selection
  was the ordered prefix of the 100-example selection.
- Local evidence directory (excluded from commit):
  `artifacts/emissary_few_shot_validation/20260907T000148960047Z/`.
- `git diff --check`: passed. Unrelated working-tree artifact deletions and local
  probes were preserved and excluded from the task commit.

The HTTP tests exercise the real low-level client, classifier, runner, artifact
writer and metrics against mocked responses. They **do not verify live API
compatibility**. No training payload, asynchronous preparation, training API
limits or job continuation is implemented or claimed tested: the required
experiment contract is absent. The three external integration tests were not run.

No paid calls, training jobs, deployments or full benchmark were executed. A local
API key exists, but training entitlement and a priced, bounded experiment flow
are unconfirmed. Observed billed cost and estimates are **unavailable**, not zero;
no provider bill was queried. Dry runs themselves made no provider requests.

## Exact cached-data procedure

The existing Hugging Face CSV loader can resolve remote file URLs even with
offline flags. The real-data dry run used its existing loader injection point
with local Arrow files; campaign sampling/splitting/definitions were unchanged.
No data was downloaded or original cache modified. Create `/tmp/issue13_cached_campaign.py`:

```python
from pathlib import Path
from datasets import Dataset
from llm_classifier_bench.datasets.huggingface import HuggingFaceClassificationDataset
from llm_classifier_bench.datasets.registry import BANKING77_SPEC
import run_banking77_scaling_benchmark_v2 as campaign

cache = Path('/home/lbiagetti/.cache/huggingface/datasets/csv/default-a38433035ea47098/0.0.0/d41f37fffd4cc4dfd07485b661c45b9863c2d0a8b0a28faa84befecfef33631a')
def loader(path, *, split, **kwargs):
    return Dataset.from_file(str(cache / f'csv-{split}.arrow'))
campaign.get_dataset = lambda name: HuggingFaceClassificationDataset(BANKING77_SPEC, loader=loader)
campaign.main()
```

Executed:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src:scripts \
  venv/bin/python /tmp/issue13_cached_campaign.py \
  --classifiers emissary --class-counts 2 --seeds 42 \
  --train-per-class 70 --test-per-class 1 \
  --min-train-per-class 70 --min-test-per-class 1 --strict-support \
  --validation-fraction .2 --emissary-shots 0 5 100 \
  --emissary-shot-unit total --dry-run \
  --output-root artifacts/emissary_few_shot_validation
```

The source profile remains `unreviewed`, as before; dry-run evidence is an
execution check, not a scientific quality result. See the
[contract blockers](emissary_contract.md) and [bounded smoke plan](emissary_smoke_plan.json)
for the remaining live acceptance work.
