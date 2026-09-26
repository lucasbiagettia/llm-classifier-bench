# Small v2 experiment — ready to plan, not running

This **2026-09-26** design replaces the earlier large matrices at the user's
request. No training, inference, recovery or scheduled execution is authorized
by preparing this plan. Verified compatible evidence is retained in `artifacts/reusable/`; superseded
run directories were moved to OS trash after an exhaustive inventory. New results
belong in `artifacts/v2_small/`. See [the cleanup audit](cleanup.json). The former large-campaign recovery command is retired.

## Size and methods

Banking77, seed **42 only**, nested cardinalities **5, 10, 15, 20**, with
**40 held-out test examples per class**. Each method/technique has four conditions:

| Classes | Test predictions |
| ---: | ---: |
| 5 | 200 |
| 10 | 400 |
| 15 | 600 |
| 20 | 800 |
| **Total per method/technique** | **2,000** |

This is 2,000 evaluated predictions across cardinalities, not 2,000 unique texts: shared
classes have the same sampled test texts at each K. All methods use paired test
IDs within each K. There is no repeated-seed or label-budget grid.

| Method / technique | Fixed labeled examples | Conditions | Current execution status |
| --- | --- | ---: | --- |
| OpenAI GPT-5 nano | **Zero-shot only; no demonstrations** | 4 | Ready |
| Emissary routing | Zero-shot | 4 | Ready |
| Emissary Quick Train | 5/class | 4 | Blocked: recorded support is UI-only |
| Emissary Projects SFT | 100/class | 4 | Blocked: previous training failed; no readiness fix recorded |
| Jev | Zero-shot | 4 | Ready; retain failures from invalid probability distributions |
| BERT fine-tuned | 50/class, two epochs | 4 | Ready on CPU, subject to time limits |
| Frozen MiniLM + logistic regression | 50/class, fixed C=1 | 4 | Ready |
| TF-IDF + logistic regression | 50/class, fixed C=1 | 4 | Ready |

**24 executable conditions**, covering **12,000 predictions**: 6,000 hosted and
6,000 local. Reuse supplies 1,700 hosted predictions, leaving at most **10,300 new
prediction calls**: 4,300 hosted and 6,000 local. Eight Emissary variant cells stay explicitly blocked and make **zero
calls and zero training jobs**. They are not replaced with another technique.
If later enabled, each adds 2,000 predictions and Projects adds training work;
that is an explicit scope extension. No 1,000-shot arm is included.

## Bounded execution

- OpenAI: at most **1,300 new** prediction attempts for the default campaign directory,
  all zero-shot. No demonstrations, warmup, automatic retries or paid pilots.
- Emissary routing: at most **1,300 new** prediction attempts plus **one** routing
  experiment creation (K=15); reuse the pinned K=5/10/20 experiments. No Projects training or deployments in the executable plan.
- Jev: at most **1,700 new** attempts, capped at 100/200/600/800 per condition.
  No retries or warmup; retain invalid responses as failures without renormalization.
- BERT: **four fits**, two epochs each, batch 16; no C/epoch sweep or second
  reference campaign. Maximum **10 minutes per condition**, **30 minutes total**
  for BERT, including loading, fitting and inference. Timeouts are incomplete
  results, not shorter successful fits. A five-second interrupt grace period can
  precede forced termination. Runtime may exceed the budget slightly during cleanup.
- Local conditions have a ten-minute wall-time limit; hosted conditions have
  thirty minutes. The saved OpenAI latency averages roughly one second per
  prediction, so K=15 needs more than ten minutes even without retries. The launcher is sequential,
  takes an execution lock and stops its active subprocess group on interrupt,
  termination or timeout. A conversational session interruption does not necessarily
  signal the launcher; its own wall-time limits remain necessary.
- Existing attempts are retained and skipped on later invocations, including
  failures. No automatic restart/refit of partial attempts. A new output directory
  is a new campaign and can spend again; these are per-campaign limits.

These are call/time limits, not a monetary ceiling. Unknown provider charges stay
unknown. Use saved usage for cost reporting, including failed attempts. Prices
come only from the recorded dated rate card, not an assertion about today's bill.

## Reuse

Eight complete zero-shot runs contribute **1,700 predictions**: OpenAI 700,
Emissary 700 and Jev 300. Reuse requires exact source hashes, class definitions
and order, test ID/text/gold identity, zero supervision and matching model/prompt
settings. All retained artifact hashes are checked. No partial failed run or old
local fit is reused. The old local training budgets/recipes differ from this plan.
Missing examples alone are sent to providers. The same command never automatically
repeats an attempted condition, even after failure or interruption.

Consolidated `predictions.jsonl` and quality metrics cover the complete paired test
set. Historical and new latency, usage, preparation and costs remain separate
with pointers to their original evidence. A complete prediction file with failed
finalization still counts as failed. Do not erase historical spending or count it
again as new spending. Selected historical runs are complete-run survivors; disclose
that reuse/availability selection when interpreting provider reliability.

## Data, training and reproducibility

Use the existing immutable Banking77 CSV revision
`57ec275d8078af65b7731c2a98be812d844a6d6b` and SHA-256 checks. Keep the existing
support pool: at least 125 source-train and 40 source-test examples per label;
42 labels qualify. Keeping 125 source candidates does **not** mean training each
model on 125 examples/class. Reserve 25/class, then choose 50/class from the
remaining pool, identically for BERT, MiniLM and TF-IDF. No validation labels are
consumed and no test scores select settings. Zero-shot methods consume no labels.

The runner records these as matched-pool conditions at budget 0 or 50, but
**zero-shot versus supervised is not an equal-supervision comparison**. The three
local methods share the same 50/class pool. Do not add a second full-training
reference matrix. Emissary's optional 5/100 arms also change mechanism and possibly
base model; they would not isolate a pure label-budget effect.

Use the unchanged `canonical_reviewed_v2` descriptions. Their review was by the
assistant, not independent human ontology validation. Previously seen test results
make this revised small campaign exploratory; it is not a new untouched test set.
The smaller BERT recipe is a scope/runtime choice, not test-based optimization.

Models: `gpt-5-nano-2025-08-07` with minimal reasoning, `jev-1.13.0`, and the
already cached pinned MiniLM/BERT snapshots in [the manifest](matrix_small.json).
BERT retains LR 2e-5, weight decay .01, max length 128 and final-epoch selection;
its existing tokenization can truncate long inputs. MiniLM's encoder is frozen.
Both LR classifiers use only C=1; there is no validation search. All local methods
run on CPU with four numeric-library threads. The installed Torch CUDA build is
incompatible with the GTX 1050; no GPU/environment changes are part of this plan.
See [the saved-evidence BERT review](bert_review.md).

## Measurements and claims

Save predictions, configurations, exact sample IDs, definition/data/model hashes,
status/errors, latency and usage/preparation ledgers. Measure accuracy, macro-F1,
ECE/adaptive ECE, log loss and Brier where outputs support them. OpenAI has no
probabilities, so calibration stays unavailable. Report median/p95 client-visible
singleton latency; each cell has fewer than 1,000 observations, so p99 is unavailable.
Inference is unwarmed, batch 1, concurrency 1; remote cache state is uncontrolled.

One seed is intentional for the small scope: **between-seed variability cannot be
estimated**. If needed, compute 95% test-sampling bootstrap intervals from saved
outputs, 2,000 draws stratified by gold class, seed 20260925, paired across methods.
They condition on this fitted model and label subset; they do not establish
robustness across label selections or training seeds. Do not pool overlapping
cardinality cohorts as independent samples. Report all failed/blocked/timed-out
conditions and avoid interpreting a BERT timeout as poor measured accuracy.

## Commands

The default command prints the 24 executable and eight blocked cells without
loading datasets/models, constructing provider clients or running inference:

```bash
PYTHONPATH=src venv/bin/python scripts/run_v2_release.py
PYTHONPATH=src venv/bin/python scripts/run_v2_release.py --only openai
```

The user can launch the prepared small matrix with:

```bash
bash scripts/run_small_experiment.sh
# Optional: select just one method
bash scripts/run_small_experiment.sh --only openai
```

Credentials are read from the existing `.env`. The wrapper uses `venv`, forces
CPU/offline cached local models and sets the repository working directory. No
command was executed against providers during preparation; nothing is scheduled.

Progress prints with timestamps every 15 seconds, including method, K, stage,
new/reused counts and elapsed time. Evidence under `artifacts/v2_small/`:

- `run.log`: overall progress, skips and errors.
- `logs/<cell>.log`: detailed provider/training output and traceback.
- `summary.json` and `execution.json`: per-condition completion/failure and counts.
- `cells/<cell>/`: full consolidated predictions/quality; `runs/new/` holds fresh
  usage, timings, costs, configuration and sample IDs.
- `manifest.json` and `execution_provenance.json`: frozen plan and executed code hashes.

Exit code **0** means all selected executable conditions completed; **2** means
at least one failed, timed out or could not run. Blocked Emissary techniques remain
visible but do not by themselves cause exit 2. Other methods continue after an
individual failure. Ctrl-C terminates the active subprocess group. Reusing the
same directory skips all previous attempts; do not change `--root` to recover a
failure, since a new directory can spend again. Bring back `summary.json` and logs
for diagnosis/reporting, including partial runs.

## Reports

The intended publication remains an extended `reports/v2/report.md` and a short
English `reports/v2/brief.md`, generated from saved evidence. No final reports or
v2.0.0 release have been produced. #15/#16 remain open until this smaller agreed
scope, its explicit limitations and reproduction instructions are delivered.
Historical spending remains separate and cannot be erased by reducing the scope.
