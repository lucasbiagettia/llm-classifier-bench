# Small v2 experiment — ready to plan, not running

This **2026-09-26** design replaces the earlier large matrices at the user's
request. No training, inference, recovery or scheduled execution is authorized
by preparing this plan. Historical results stay in `artifacts/v2_release/`;
new results belong in `artifacts/v2_small/` and must not be silently pooled with
those exploratory runs. The former large-campaign recovery command is retired.

## Size and methods

Banking77, seed **42 only**, nested cardinalities **5, 10, 15, 20**, with
**30 held-out test examples per class**. Each method/technique has four conditions:

| Classes | Test predictions |
| ---: | ---: |
| 5 | 150 |
| 10 | 300 |
| 15 | 450 |
| 20 | 600 |
| **Total per method/technique** | **1,500** |

This is 1,500 evaluation calls across cardinalities, not 1,500 unique texts: shared
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

**24 executable conditions**, up to **9,000 evaluation calls**: 4,500 hosted and
4,500 local. Eight Emissary variant cells stay explicitly blocked and make **zero
calls and zero training jobs**. They are not replaced with another technique.
If later enabled, each adds 1,500 predictions and Projects adds training work;
that is an explicit scope extension. No 1,000-shot arm is included.

## Bounded execution

- OpenAI: at most 1,500 prediction attempts for the default campaign directory,
  all zero-shot. No demonstrations, warmup, automatic retries or paid pilots.
- Emissary routing: at most 1,500 prediction attempts plus at most four routing
  experiment creations. No Projects training or deployments in the executable plan.
- Jev: at most 1,500 attempts, capped separately at 150/300/450/600 per condition.
  No retries or warmup; retain invalid responses as failures without renormalization.
- BERT: **four fits**, two epochs each, batch 16; no C/epoch sweep or second
  reference campaign. Maximum **10 minutes per condition**, **30 minutes total**
  for BERT, including loading, fitting and inference. Timeouts are incomplete
  results, not shorter successful fits. A five-second interrupt grace period can
  precede forced termination. Runtime may exceed the budget slightly during cleanup.
- All conditions have a ten-minute wall-time limit. The launcher is sequential,
  takes an execution lock and stops its active subprocess group on interrupt,
  termination or timeout. A conversational session interruption does not necessarily
  signal the launcher; its own wall-time limits remain necessary.
- Existing attempts are retained and skipped on later invocations, including
  failures. No automatic restart/refit of partial attempts. A new output directory
  is a new campaign and can spend again; these are per-campaign limits.

These are call/time limits, not a monetary ceiling. Unknown provider charges stay
unknown. Use saved usage for cost reporting, including failed attempts. Prices
come only from the recorded dated rate card, not an assertion about today's bill.

## Data, training and reproducibility

Use the existing immutable Banking77 CSV revision
`57ec275d8078af65b7731c2a98be812d844a6d6b` and SHA-256 checks. Keep the existing
support pool: at least 125 source-train and 30 source-test examples per label;
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

After an explicit new instruction to run, execute one method or the small matrix:

```bash
PYTHONPATH=src venv/bin/python scripts/run_v2_release.py --execute --only openai
PYTHONPATH=src venv/bin/python scripts/run_v2_release.py --execute
```

The same output directory skips already attempted cells. Do not run either live
command as part of preparing this design. Nothing is scheduled.

## Reports

The intended publication remains an extended `reports/v2/report.md` and a short
English `reports/v2/brief.md`, generated from saved evidence. No final reports or
v2.0.0 release have been produced. #15/#16 remain open until this smaller agreed
scope, its explicit limitations and reproduction instructions are delivered.
Historical spending remains separate and cannot be erased by reducing the scope.
