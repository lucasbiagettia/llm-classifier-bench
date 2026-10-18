# Hard-dataset pilot preparation

This adds preparation and execution commands for ImplicitHate (all six requested
fine labels) and ANLI R3 (all three labels). **No paid runs or real-dataset model
training have been performed.** Preparation counts and cost projections below are
not experimental performance results. AG News was already implemented; the new
entry point also accepts `--datasets ag_news` and uses all four labels.

## Evidence used to choose the pilot

The source of truth is the checked-in `reports/v2/manifest.json` and the local raw
artifacts, rather than the old campaign's command-line defaults:

- Banking77: K=5/10/15/20, seed 42, **40 held-out test examples per class**, hence
  200/400/600/800 evaluation examples (2,000 across the four conditions per method).
- Local supervised classifiers: **20/50/100 fit examples per class**, not 100 total.
  At K=5 those are 100/250/500 fit examples; at K=20, 400/1,000/2,000.
- Source candidates were 125/class, with 25/class reserved for validation. The
  published matched-budget runs **consumed zero validation labels**. For example,
  `artifacts/v2_completion/cells/emissary-qwen__s42__k15__b100/runs/new/config.json`
  records 1,500 fit IDs, zero validation IDs and 600 test IDs. A training setting
  is not evidence that those examples were all used for fitting.
- OpenAI `gpt-5-nano-2025-08-07` (`minimal` reasoning), Jev `jev-1.13.0` and Emissary
  routing were zero-shot. Projects SFT used `Qwen3-4B-Base` and
  `Llama-3.2-1B-Instruct`, one epoch, 20/100 fit examples/class. BERT used
  `google-bert/bert-base-uncased`, two epochs, batch 16, learning rate 2e-5,
  weight decay .01 and length 128. Frozen MiniLM and TF-IDF used LR C=1.
- The latest supplements use `gpt-6-luna` Decisions on the same 2,000 test cases
  (ten refusals explicitly accounted for); Perplexity Decisions
  `pplx-decider-v1.1-27b` is also implemented in the current code. Generative
  OpenAI in-context classification exists, although it was not one of those
  published zero-shot conditions.
- Historical inference costs are usage-ledger/rate-card estimates, with unknown
  totals retained as unavailable (including the interrupted Decisions condition).
  They are not universal per-example prices; new input lengths and methods change
  token use. No missing price is interpreted as zero.

For a concrete historical cost reference, the K=5, 20/class (or zero-shot) raw
`cost_report.json` files under the condition `evidence` paths in
`reports/v2/results.json` give the following **inference-only estimates for 200
predictions**: OpenAI USD 0.00189695, Jev USD 0.001754508, TF-IDF USD
0.000007462, MiniLM USD 0.000118561 and BERT USD 0.000454986. The local values
are cloud-equivalent projections of active inference time, not bills; training
is excluded. All three Emissary totals are unavailable. These saved observations
are distinct from the deliberately generous token allowances in the new plan.

The pilot selects the **smallest historical supervised budget, 20/class**, and
retains 40 test examples/class. This gives 120 fit + 240 test examples for
ImplicitHate and 60 fit + 120 test examples for ANLI. The former is close to the
smallest Banking77 condition; the latter is smaller because there are only three
classes. There is one seed (42), no label-cardinality sweep, no warmup, batch size
one and no automatic retry. This is a small pilot, not an estimate of variation
across seeds. Increase the sizes or repeat a separately prepared seed when needed.

## Sources, labels, explanations and partitions

### ImplicitHate

Sources: [Hugging Face](https://huggingface.co/datasets/SALT-NLP/ImplicitHate),
[official repository](https://github.com/SALT-NLP/implicit-hate), and
[Latent Hatred paper](https://aclanthology.org/2021.emnlp-main.29/).
The CSV is pinned at HF revision
`57d34fad1cf7d4d9e71774ea17c4c1a7f57af8d4`. Its columns are `post`,
`implicit_class`, `extra_implicit_class`, with one source `train` partition.

Only the primary `implicit_class` is the target. The six classes are
`white_grievance`, `incitement`, `inferiority`, `irony`, `stereotypical`,
`threatening`. All 80 rows whose primary label is `other` are explicitly excluded.
Secondary labels remain annotation metadata, without expanding the target space.
Unknown primary labels cause an error.

Identical posts are a single scoring/sampling unit containing all their source
annotations. Conflicting primary labels exclude the entire group and are counted.
The pinned source has 6,346 rows, 6,266 after excluding `other`, and 6,265 retained
unique posts (one duplicate, no conflicting groups). Per-class groups are sorted
by SHA-256 input ID and shuffled with an independent seed string, then split at
floor(.6*N) and floor(.8*N). The resulting 60/20/20 **synthetic**, stratified
train/dev/test partitions contain 3,756/1,253/1,256 examples. They are not claimed
as the paper's official splits. Partition seed 42 is fixed independently of the
pilot sampling seed, so changing sample budgets or seeds never moves the holdout.

The HF CSV contains no target/implied-message annotations: current coverage is
**0/6,265** for both fields, not assumed complete from the paper's description.
If the official stage-3 TSV is available locally, use `--hate-annotations PATH`.
The file must have `post` and may contain `target` and `implied_statement`.
Only exact original post matches are joined, keeping every matching annotation
and the file SHA-256. Unmatched posts are counted, never fuzzy-matched or filled
with generated explanations. Native explanation fields are also retained if
present in input rows. Multiple explanations remain together under `annotations`.
The official data-access instructions include a survey; the public HF mirror has
no explicit license declaration in its card. This change does not resolve those
access/license questions or redistribute the corpus in Git.

### ANLI R3

Sources: [Hugging Face](https://huggingface.co/datasets/facebook/anli) and
[official repository](https://github.com/facebookresearch/anli). Parquet files
are pinned at HF revision `8e4813d81f46d313dac7892e1c28076917cfcdf9`.
Only `train_r3`, `dev_r3`, `test_r3` are used; no mixing of rounds.
Numeric labels 0/1/2 map to `entailment`/`neutral`/`contradiction`.

Every classifier sees exactly this raw text format:

```text
Premise: {premise}
Hypothesis: {hypothesis}
```

`uid`, original numeric label, `reason`, and original fields stay in annotation
metadata. Original counts are 100,459/1,200/1,200. Grouping repeated input pairs
retains 100,361 training units. There is one exact pair shared by train and dev,
none between train/test or dev/test. Holdout priority is test > dev > train:
exclude duplicate lower-priority candidates and attach their annotations to the
existing higher-priority unit. Unique examples never change official partitions;
test is unchanged. The resulting train/dev/test counts are 100,360/1,200/1,200.
All exclusions and original split provenance are recorded. This necessary change
is explicit rather than silently sampling around cross-split duplicates.

On the pinned, grouped source, nonempty `reason` coverage is 13,374/100,360 train,
1,200/1,200 dev and 1,200/1,200 test. Empty reasons remain missing. The official
repository documents the dataset as CC BY-NC 4.0; consult its license for use.

Both tasks retain explanations only for analysis. `LabeledExample.as_input()`
forwards text and allowlisted provenance, stripping labels, explanations, targets
and arbitrary annotation metadata. Frozen snapshots include gold metadata but
are never sent directly to inference. Tests verify the actual inputs delivered
through the runner for all eleven methods.

## Matrix and configuration

Defaults live in `configs/hard_datasets_pilot.json`; all classes are used. Class
rubrics are frozen source constants in `datasets/hard_tasks.py`, paraphrasing the
paper taxonomy and the NLI truth conditions above. Every method receives the same
rubrics and the same ordered test inputs. Snapshots persist the exact definitions,
texts, IDs, annotations and SHA-256 hashes. Definitions are not generated by an API.

| Variant | Mode | Fit labels per task (hate / ANLI) | Context labels | Validation consumed |
| --- | --- | ---: | ---: | ---: |
| TF-IDF + LR | supervised | 120 / 60 | 0 | 0 |
| MiniLM + LR | frozen embeddings, supervised LR | 120 / 60 | 0 | 0 |
| BERT | fine-tuned | 120 / 60 | 0 | 0 |
| OpenAI | zero-shot | 0 | 0 | 0 |
| OpenAI context | in-context | 0 | 5 total/task | 0 |
| Jev | zero-shot | 0 | 0 | 0 |
| OpenAI Decisions | zero-shot | 0 | 0 | 0 |
| Perplexity Decisions | zero-shot | 0 | 0 | 0 |
| Emissary routing | zero-shot | 0 | 0 | 0 |
| Emissary Qwen SFT | Projects SFT | 120 / 60 | 0 | 0 |
| Emissary Llama SFT | Projects SFT | 120 / 60 | 0 | 0 |

Models match those listed above. BERT length is increased to **512**, with the
same two epochs and other historical settings, because NLI pairs are longer.
MiniLM retains its encoder's native truncation (normally 256 tokens). Both receive
the same formatted raw string; this is not a specialized pair-encoder baseline.
Long inputs can lose trailing hypothesis content under truncation: inspect length
metadata before interpreting comparisons. No tokenizer/model download is needed
for planning, so exact model-token truncation counts are not claimed. On the
prepared default test samples, raw character ranges are 13–801 (hate) and 187–727
(ANLI). Classifier artifacts preserve their tokenization settings.

LR uses fixed C=1, matching the published runs; optional validation is an extra,
separately counted label budget. Use a custom config C grid to enable meaningful
LR selection. The backward-compatible `DatasetBundle.validation` field lets the
runner consume official dev data without mixing it into train. `None` keeps the
old train-splitting behavior, while `()` explicitly uses no validation. This pilot
uses `--validation-per-class 0`. With a positive count, local models consume a
stratified official/synthetic dev subset; SFT receives that disjoint subset for
checks but its adapter consumes **zero** validation labels (last checkpoint).

Five total in-context examples are drawn reproducibly from the same 20/class fit
pool. With six classes, five examples cannot cover every class. This is a separate
low-budget mode, not a matched 20/class training comparison. Supervised methods
use identical fit members. No explanation is used as a demonstration or prompt.
Emissary routing's provider-managed model differs from the two explicitly chosen
SFT bases; that model/mechanism confound remains part of the comparison.

## Preparation and execution

From the repository root (use `venv/bin/python` here, or your active environment):

```bash
# Default: prepare BOTH datasets and all variants; no fitting or paid API calls.
PYTHONPATH=src venv/bin/python scripts/run_hard_datasets.py \
  --dry-run --output-dir artifacts/hard_pilot

# Optional: incorporate the locally obtained full-corpus annotations.
PYTHONPATH=src venv/bin/python scripts/run_hard_datasets.py \
  --hate-annotations /path/to/implicit_hate_v1_stg3_posts.tsv \
  --output-dir artifacts/hard_pilot_with_explanations

# Change budgets or seed in a NEW plan. No silent support backoff.
PYTHONPATH=src venv/bin/python scripts/run_hard_datasets.py \
  --train-per-class 50 --test-per-class 40 --seed 43 \
  --output-dir artifacts/hard_pilot_50

# AG News is already supported, with all FOUR labels.
PYTHONPATH=src venv/bin/python scripts/run_hard_datasets.py \
  --datasets ag_news --methods tfidf --output-dir artifacts/ag_news_all_labels
```

Preparing may download public dataset files, but creates no API client, remote
resource, transformer model or trained classifier. It writes `plan.json`, a plan
checksum, a merged frozen `pricing.json`, and one labeled snapshot per dataset.
The plan records source revisions/fingerprints, source and selected per-class
counts, coverage of each explanation field, all condition budgets and blockers,
context checks, input lengths, test IDs, calls, tokens and priced estimates.
Source partitions are audited for overlapping IDs and exact content before any
subsampling. Inadequate support fails rather than reducing the budget.

To prepare executable SFT cells, supply an **existing** Projects workspace with
`--emissary-project-id PROJECT_ID` or the config's `emissary.project_id`. Until then
all four SFT cells remain visibly blocked. The other variants can be prepared
separately with `--methods`; no default variant disappears silently.

The following commands are provided for later use; they have **not** been run on
these datasets:

```bash
# Optional local-only campaign; BERT/MiniLM can download models and train.
PYTHONPATH=src venv/bin/python scripts/run_hard_datasets.py \
  --methods tfidf sentence-transformer bert --output-dir artifacts/hard_local
PYTHONPATH=src venv/bin/python scripts/run_hard_datasets.py \
  --execute --plan artifacts/hard_local/plan.json

# Full hosted matrix: PREPARE with an existing project first.
PYTHONPATH=src venv/bin/python scripts/run_hard_datasets.py \
  --emissary-project-id PROJECT_ID --output-dir artifacts/hard_full
PYTHONPATH=src venv/bin/python scripts/run_hard_datasets.py \
  --execute --plan artifacts/hard_full/plan.json \
  --max-api-calls 2880 --max-training-jobs 4 \
  --max-estimated-usd 2 --allow-unpriced
```

Execution uses the frozen snapshots and config, never downloads/resamples dataset
inputs, and verifies snapshot, pricing and plan checksums before work. New run
folders under `executions/` preserve the common runner's config, fit metadata,
predictions, metrics, operational timings, usage and preparation reports. A
summary CSV/JSON is updated after every attempted condition. Runs stop on the
first error/refusal; partial predictions stay saved. There is no automatic resume:
rerunning execution starts a new campaign and may repeat paid work.

The call cap bounds **inference attempts** across the campaign, with retries and
warmup disabled. It excludes API resource creation and bounded polling. The SFT
job cap defaults to zero and must cover the four planned jobs explicitly. No
provider HTTP requests are made if a plan is blocked or exceeds a cap. The USD
option guards the **priced estimate only**, not the final bill. No global hard
dollar cap is exposed by the existing adapters. Missing Emissary tariffs mean a
full campaign cannot have an honest total dollar estimate; `--allow-unpriced`
explicitly acknowledges that before later execution.

## Cost projection and metrics

The real-data preparation validated 22 cells and 360 held-out inputs across tasks:
3,960 predictions over all methods, of which **2,880 are hosted inference calls**.
There are four planned SFT jobs. Each task's five ordinary zero-shot variants,
one in-context variant, and two SFT variants account for eight hosted calls/input.

The versioned September/October rate cards are copied into the plan. They are
historical estimates, **not a claim that current account pricing was verified**.
Input token proxies count UTF-8 bytes of text, definitions and demonstrations plus
2,048 framing bytes/request, with no cache discount. Generative OpenAI reserves
4,096 output tokens/request; Decisions and Jev output are priced at zero under
their stored tariffs. These are approximate planning allowances, not provider
token counts, actual spend or guaranteed upper bounds.

| Priced variant | ImplicitHate estimate, USD | ANLI R3 estimate, USD |
| --- | ---: | ---: |
| OpenAI zero-shot | 0.430017 | 0.213979 |
| OpenAI context | 0.437937 | 0.227125 |
| Jev | 0.030913 | 0.014591 |
| OpenAI Decisions | 0.073602 | 0.034741 |
| Perplexity Decisions | 0.014720 | 0.006948 |

Priced subtotal: **USD 1.484574**. Emissary routing, SFT inference/training/deployment
and local compute are **unavailable**, not free. For those, missing information is
provider/account tariffs or returned charges, training resource usage, deployment
allocation time and a matching local hardware rate. Cost equivalence to historical
runs is therefore only a design goal, not a measured conclusion.

The existing runner computes accuracy, macro-F1, top-label ECE, adaptive ECE,
multiclass log loss, Brier score, mean/P50/P95/P99 latency and inference cost per
1,000 predictions, with preparation and throughput reports alongside. Generative
OpenAI (including context mode) has no probability outputs, so calibration/log-loss/
Brier stay unavailable. P99 requires 1,000 timing observations and is unavailable
for each default condition. No missing scores are replaced with zero. Keep failed
or refused cases explicit; do not describe a partial cohort as a complete result.

## Validation and scope

Offline regression tests cover labels, unknown-label rejection, official and
synthetic partitions, duplicate/conflicting annotations, supplemental exact joins,
explanation stripping, reproducibility, all-method shared inputs, integrity/call/
job/estimate guards, and real TF-IDF execution on a small **synthetic fixture**.
The actual pinned HF loaders and the complete preparation matrix were also run,
with no training or paid inference. The compact preparation audit is versioned at
`reports/hard_datasets/preparation.json`; full prepared snapshots stay under ignored
`artifacts/` and can be reproduced using the frozen revisions and default command.

This is a partial contribution to [issue #9](https://github.com/lucasbiagettia/llm-classifier-bench/issues/9).
It follows the datasets explicitly requested here; it does not claim to complete
the issue's SELU candidate review or all future task-selection work.

Validation command: `PYTHONPATH=src venv/bin/python -m pytest -m 'not integration' -q`
— 495 passed, 3 integration tests deselected. `git diff --check` also passed.
Prepared CSV/Parquet URL resolution still requires network access even with a
populated HF cache; the frozen-plan execution path does not resolve dataset URLs.
