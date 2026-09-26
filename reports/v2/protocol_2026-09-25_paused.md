> Historical design, superseded by [the small experiment](protocol.md). Do not execute its commands.

# V2 experiment design — paused, revised 2026-09-25

**Execution is stopped. No calls, training, automatic restart or scheduled run.**
This is the plan to resume only on a new instruction. It supersedes the
[original frozen scope](protocol_2026-09-25_original.md) by limiting OpenAI to
zero-shot. Preserve the original artifacts and exclude OpenAI few-shot from the
v2 comparison. The general adapter still supports in-context experiments, but the
release launcher and recovery command prohibit them.

## Common data and measurement

| Dimension | Frozen choice |
| --- | --- |
| Dataset | Banking77, immutable revision `57ec275d8078af65b7731c2a98be812d844a6d6b` |
| Cardinalities | 5, 10, 20, 25 nested labels per seed |
| Seeds | 42, 43, 44 |
| Source training | 125 examples/class; strict support, no automatic reduction |
| Eligible population | 42 of 77 labels have enough source training examples |
| Reserved validation | 25 examples/class, drawn only from source training |
| Final test | 20 original held-out examples/class; same IDs across methods/budgets |
| Class definitions | Frozen `canonical_reviewed_v2`; assistant review, not independent human validation |
| Inference | Batch 1, concurrency 1, zero warmup, no hidden retries |
| Local compute | CPU, four numeric-library threads; cached pinned model snapshots |
| Calibration | ECE, adaptive ECE, log loss, multiclass Brier where probabilities exist |
| Quality uncertainty | Accuracy/macro-F1; 2,000 gold-class-stratified paired bootstrap draws, 95% percentile intervals |
| Seed variation | Individual seeds, mean/sample SD/range; no pooling overlapping test IDs as independent observations |
| Operations | Client-visible latency and failure coverage; p99 unavailable below 1,000 calls/cell |
| Cost | Saved usage and dated rates; CPU estimates are illustrative, Emissary price remains unknown |

## Method matrix

All methods use the same cardinalities, seeds, test IDs and frozen definitions.
Shots mean distinct labeled examples **per class**.

| Method | Zero-shot | 5/class matched | 100/class matched | Supervised reference |
| --- | --- | --- | --- | --- |
| OpenAI `gpt-5-nano-2025-08-07`, minimal | Yes, **only OpenAI condition** | Excluded by user | Excluded by user | Reuse zero-shot control |
| Jev `jev-1.13.0` | Yes | Unsupported | Unsupported | Reuse zero-shot control |
| Emissary routing | Yes | Quick Train unavailable in API | Projects SFT blocked by failed provider training | Reuse zero-shot control |
| TF-IDF + LR | Unsupported | Yes | Yes | 100 fit + 25 validation/class |
| Frozen MiniLM + LR | Unsupported | Yes | Yes | 100 fit + 25 validation/class |
| BERT fine-tuned | Unsupported | Yes | Yes | 100 fit + 25 validation/class |

Matched runs consume zero validation labels. The reserved validation partition is
withheld. Reference runs use those validation labels to select C/epoch. Keep the
regimes in separate tables. Zero-shot controls are reused between views, never
counted as independent runs or rebilled. Do not call a zero-shot versus supervised
comparison “equal labeled budget.”

There are **228 requested cells**: 192 matched/support-accounting cells and 36
supervised references. Of these, 144 are eligible to execute and 84 explicitly
unsupported. Another 36 reference-table entries reuse the zero-shot controls.
The 24 historical OpenAI 5/100-shot cells are excluded from the revised design;
their artifacts and costs remain in the historical accounting.

## State at stop and work remaining

- 115 completed, 78 unsupported, 9 failed/interrupted, 26 unstarted: 228 total.
- Seven Jev cells failed probability-sum validation. Keep them failed under the
  frozen criterion; do not silently renormalize or repeatedly sample until valid.
- One OpenAI zero-shot cell remains: seed 44, K=20, 400 test inputs, no saved
  predictions. Its original failure was a token rate limit during the former
  few-shot campaign. Continuation is restricted to this zero-shot scope and
  retains original failure/usage evidence.
- BERT has 15 completed supported cells. One 5-shot cell (seed 43, K=10) was
  interrupted during fitting, with zero predictions. Retain that attempt; an
  explicit restart can refit it without duplicating any completed predictions.
  Twenty other supported BERT cells and six unsupported accounting cells have
  not started. See the [offline BERT review](bert_review.md).
- The old OpenAI recovery did **not** submit calls: it stopped when the dry-run's
  derived definition file already existed. The corrected planner uses temporary
  derived definitions and is idempotent; it still needs live validation tomorrow.

## Fixed training policy and limitations

BERT keeps three epochs, batch 16, LR 2e-5, weight decay .01, max length 128.
Matched budgets select the final epoch; reference selects minimum validation loss.
MiniLM and TF-IDF retain existing C policies. Do not retune after seeing test scores.
Five-shot BERT gets only 6/12/21/24 optimizer steps at K=5/10/20/25 under this
fixed-epoch policy; this is a limitation of that training recipe, not proof of
BERT's best attainable few-shot performance. Any alternative recipe is a separate,
predeclared experiment and cannot replace these results post hoc.

The installed CUDA build lacks GTX 1050 kernels. CPU execution is intentional and
must remain fixed when reusing existing latency measurements. Do not install a new
Torch build or switch hardware silently. The adapter's automatic device selection
checks CUDA availability, which by itself does not establish kernel compatibility;
this is a robustness gap, documented in the BERT review.

Cardinality curves change label composition. A shared-test-cohort contrast can
hold the low-cardinality examples fixed, but not the supervised fit pool across K.
Intervals are pointwise/exploratory with no multiplicity correction. Seed effects
combine class/example selection and training randomness; they are not pure training
variance. Missing seed results must remain visible, without unqualified averages.

## Tomorrow's commands

Read [the handoff](../../.local/handoffs/v2-paused.md) locally for the stop inventory.
The checked-in [revised manifest](matrix_next.json) contains the exact arguments,
source/model hashes and cell identities. The original manifest remains historical.

Plan only: no inference, training or provider clients:

```bash
PYTHONPATH=src venv/bin/python scripts/run_v2_release.py --only bert --restart-interrupted
PYTHONPATH=src venv/bin/python scripts/resume_v2_openai.py
```

Only after a new instruction to run (these are **not** scheduled):

```bash
PYTHONPATH=src venv/bin/python scripts/run_v2_release.py --execute --only bert --restart-interrupted
PYTHONPATH=src venv/bin/python scripts/resume_v2_openai.py --execute
```

Run sequentially to avoid adding workload contention to latency measurements.
Completed and unsupported cells are skipped. Other failed cells are retained;
partial outputs must never be restarted from scratch. General trained-model
checkpoint resume remains unimplemented; the interrupted BERT cell has no output
predictions and must repeat its incomplete fitting work explicitly.

## Publication and open issues

Prepare `reports/v2/report.md` (extended English report) and `reports/v2/brief.md`
(short English brief for Tanmay), plus tables, plots and reproducible retrieval
instructions. These reports are **not yet generated**, and no release is published.
All tables/plots must replay saved predictions and ledgers without inference.
Include excluded historical spending, Jev failures, Emissary training limitations,
review status, CPU conditions and uncertainty. Cost totals with unavailable
provider charges must not be presented as audited complete spending.

Issues #15 and #16 remain open. Verify their acceptance criteria before publishing
`v2.0.0`; do not claim generic resume, a complete matrix, or a confirmed spending
ceiling when the retained evidence does not establish them.
