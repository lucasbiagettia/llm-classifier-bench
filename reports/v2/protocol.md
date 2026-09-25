# Frozen v2 campaign protocol — 2026-09-25

Status: frozen before final evaluation. This supplements the repository's
[measurement protocol](../../docs/experimental_protocol_v2.md).

Banking77, nested 5/10/20/25-label prefixes, seeds 42/43/44, 125 source-train and
20 held-out test examples per class. Strict support: 42 of 77 labels qualify;
results describe this support-filtered population. Reserve 20% (25/class) from
source training. No test-driven description, hyperparameter or subset changes.
The immutable CSV revision and hashes are in the campaign `source.json`.

Six methods: TF-IDF LR, frozen MiniLM LR, BERT, GPT-5 nano, Emissary routing,
and Jev. Matched budgets are 0/5/100 labeled examples **per class**, with zero
consumed validation labels. The separate full-training reference uses 100 fit
and 25 validation examples/class for the three local supervised methods.
Zero-shot controls are reused in the reference view: the exact same test IDs,
definitions and predictions, zero consumed labels, and no extra observations.
The regimes are never pooled. There are 252 physical cells (216 matched, 36
supervised reference), plus 36 reused reference controls.

Emissary 5-shot Quick Train is UI-only according to the handoff and cannot be
substituted with Projects SFT. Projects SFT remains unavailable after the recorded
training failure. No new grid of training jobs is submitted without a successful
readiness check; nonzero Emissary cells remain explicitly unsupported with this
limitation. Jev nonzero matched budgets and local zero-label fits are unsupported.
OpenAI prompts must pass the existing conservative context preflight; do not
truncate demonstrations to make a cell fit.

Models: `jev-1.13.0`, `gpt-5-nano-2025-08-07` (minimal reasoning), and the recorded
local MiniLM/BERT snapshots. BERT: three epochs, batch 16, LR 2e-5, weight decay
.01, max length 128; validation selects the epoch only in the reference regime.
LR methods retain their existing defaults and selection rules. BERT's tokenizer
uses its existing truncation policy. All inference calls use batch 1, concurrency
1, no warmup, no retries. This is unwarmed client-visible latency; p99 is
unavailable below 1,000 observations per cell. Client location is unspecified;
the workstation's timezone is not evidence of network location.

The installed CUDA build cannot execute on the GTX 1050. All local models run
on CPU, four numeric-library threads. Disk model caches are reused; provider
cache state is uncontrolled. The model files are pinned to cache snapshots and
recorded with retrieval revisions. The selected 2026-09-25 rate card supplies
estimates, not invoices; CPU prices are illustrative cloud equivalents and
Emissary prices remain unavailable. User instruction on 2026-09-25 authorizes
the defined campaign without further price approval. No claim of an audited
sub-USD-100 total can be made while Emissary charges are unavailable.

The `canonical_reviewed_v2` profile was reviewed by the Codex assistant against
all 77 names before evaluation. Four overly narrow/ambiguous descriptions were
corrected using names alone. No source examples were consulted for this review.
This is not independent human ontology validation; overlapping labels remain a
limitation. The prior enriched profile is preserved unchanged.

Primary outcomes: accuracy and macro-F1. Report every seed individually, plus
mean, sample standard deviation and range across the three seeds. These seeds
vary class selection, source example selection and stochastic training together;
they do not isolate training randomness. Overlapping test cohorts must not be
concatenated as independent samples.

Test uncertainty: 2,000 bootstrap draws, seed 20260925, percentile 95% intervals,
resample within gold class preserving class sizes and the frozen label inventory.
Hold fitted model and campaign seed fixed. For within-condition method/budget
contrasts use identical resampled test IDs; effects are first minus second.
Intervals are pointwise/exploratory, without multiplicity correction. Report
cross-seed variability separately, not as a three-seed estimate of population
uncertainty. Calibration is descriptive ECE/adaptive ECE/log loss/Brier with the
existing formulas and availability rules; no fabricated OpenAI probabilities.

Cardinality curves change the evaluated label mixture. Additionally compare
predictions on the shared low-cardinality test cohort where saved IDs/text agree;
this fixes test examples but does not fix the supervised fit pool across K.
A deterioration is an empirical result to test, not a required conclusion.

Retain each status, error, prediction, timing, cost/preparation ledger, configuration,
source hash and code revision. Generate extended report, figures and English brief
from saved evidence without new inference. Terminal cells are skipped on restart.
Interrupted partial cells must preserve prior predictions and cannot be restarted
from scratch. General partial-cell recovery is a release gate until validated.
