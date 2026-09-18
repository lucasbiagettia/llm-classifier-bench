# v2 experimental protocol and release scope — issue #6

Status: **draft for owner decisions**, 2026-09-15. This document specifies the
experiment; it does not launch it. Proposed settings below are not claims that
the current campaign runner already implements them. Resolve the dependency
table before freezing a final evaluation manifest.

## 1. Objective and release boundary

Compare closed-set classification quality, probability calibration, inference
latency, and cost as label count and labeled fit budget change. Report results
by supervision regime, dataset, label count, fit budget, and seed. A supervised
win over zero-shot does not establish a like-for-like improvement.

v2 extends the existing runner, adapters, metrics, and saved artifacts. Deliver
this protocol, a frozen campaign manifest, prediction/metric artifacts, and a
report with uncertainty and failure coverage. New abstraction layers, a custom
dataset, and a complete SELU evaluation are outside the release scope.

### Owner decisions still open

| Decision | Working proposal | Status |
| --- | --- | --- |
| Shot values | 5 and 100 labeled fit examples; zero-shot controls | Requested values retained |
| Shot unit | **Total**, following the September Emissary smoke; `per_class` is an explicit alternative below | Final-v2 confirmation pending; smoke confirmation is not a final protocol decision |
| Datasets | Banking77 for final evaluation; AG News for technical smoke only | Confirmation pending; second final dataset not selected |
| Total incremental spend | Proposed USD 50, including pilot, retries, training and serving | Owner amount pending; not authorized |
| Pilot allocation | Proposed USD 5 within the total, not in addition | Owner amount pending; not authorized |

No paid evaluation starts while budget decisions remain unresolved. The amounts
above are planning placeholders, not estimates of provider prices or evidence
that the full matrix fits the budget.

## 2. Implementation and artifact inventory

Reviewed at main commit `241933e`. Issue #7's separate audit is at
[`a1c2139`](https://github.com/lucasbiagettia/llm-classifier-bench/blob/a1c2139/docs/calibration_audit_v2.md).

| Component | Implemented evidence | v2 role / remaining limit |
| --- | --- | --- |
| TF-IDF + LR | `classifiers/tfidf.py`; real cached Banking77 validation and replay in [TFIDF_VALIDATION.md](../TFIDF_VALIDATION.md) | Include explicitly; omitted from the campaign's default classifier list |
| MiniLM + LR | Frozen `all-MiniLM-L6-v2`, validation accuracy selects C; four historical scaling runs | Include; merge #7 artifact validation and candidate-score metadata |
| BERT | `google-bert/bert-base-uncased`, supervised fine-tuning; validation loss selects epoch | Include |
| OpenAI | Structured label-only zero-shot adapter; historical scaling runs | Include; no comparable confidence/probability output; in-context few-shot is not implemented |
| Emissary routing | Zero-shot class names/descriptions and inference | Include as a zero-shot control |
| Emissary Projects SFT | Explicit separate base-model training, dataset/job/deployment persistence, offline tests | Conditional inclusion; live training acceptance remains unresolved |
| Datasets | Banking77 and AG News registry adapters | Banking77 scaling; AG News has four labels and cannot support K=5/10/20/25 |
| Sampling | Support-filtered nested class prefixes, seeded per-class source samples, persisted split IDs | Current fit/validation split is not stable across K; shared fit-budget selection across local classifiers is missing |
| Metrics | Accuracy, macro-F1, fixed/adaptive ECE, log loss, Brier, mean/p50/p99 latency, cost reducers | Bootstrap, failure-aware reporting and end-to-end cost attribution still needed |

The historical Banking77 campaign
[`20260804T014841Z`](../artifacts/benchmark_runs/20260804T014841Z/campaign.json)
contains 16 runs: four classifiers × K={5,10,20,25}, seed 42. Its supervised
budget was **100 source-training examples per class, split into 80 fit + 20
validation**, with 20 test examples per class. It is not a 100-fit-shot experiment.
The source-support filter admitted 66 of 77 labels. Cost was unavailable.

The Emissary 0/5/100 smoke used **total** fit examples at K=2. Uploaded datasets
profiled successfully; the September 7 evidence reports a payment-method error
before training. See [validation](emissary_validation.md),
[contract](emissary_contract.md), and [smoke plan](emissary_smoke_plan.json).
These are dated observations; account readiness and pricing must be checked
again before execution.

## 3. Proposed evaluation matrix and shot semantics

Main dataset: Banking77. Final K={5,10,20,25}; repetitions
`seeds=[42,43,44,45,46]`. Five seeds are a pragmatic initial scope; publish
individual-seed values and do not claim broad dataset generality.

| Regime | Classifiers | Fit examples | Inference-context examples |
| --- | --- | --- | --- |
| Zero-shot controls | OpenAI, Emissary routing | 0; no validation consumed | 0 |
| Fit-budget curve | TF-IDF + LR, MiniLM + LR, BERT; Emissary Projects if ready | 5 and 100, using the confirmed unit | 0 |
| Historical-support reference | TF-IDF + LR, MiniLM + LR, BERT | 80 per class, plus 20 validation per class | 0 |

Run each zero-shot control once per K/seed, then reuse that result across budget
plots. Repeated plotting is not an independent run. The historical-support
reference is a separate condition; it is never relabeled as 100-shot.

**A shot means an example actually used to fit parameters**, not an example
in an inference prompt and not an example reserved for validation. Track
`fit_examples_used`, `validation_examples_used`, `context_examples_used`, their
per-class counts and IDs, and `total_labeled_examples_used` for every run.
Do not use the current `--train-per-class` flag as a synonym for fit shots.

For the working `total` proposal, S=5 or 100 is spread by balanced round-robin
across the K labels. S=100 gives 20/10/5/4 fit examples per class at K=5/10/20/25.
S=5 covers every class only at K=5. **Mark S=5, K>5 supervised conditions
`infeasible_insufficient_class_coverage` before training**, including Projects
SFT. Do not oversample, add examples, shrink the requested class set, or invent
zero probabilities to make them run. A future partial-coverage study is outside
this release. This leaves 25 feasible K/budget/seed cells per supervised method.

If the owner chooses `per_class`, S means S for every class and the fit total is
S×K. All requested cells then have class coverage, subject to dataset/provider
support. Update the matrix and eligible label pool before freezing the manifest;
do not mix units in one curve.

### Validation budget is an additional labeled resource

Reserve a fixed **20 validation examples per selected class** from source train,
disjoint from all fit budgets. Local supervised methods use this set for model
selection. Thus a local total-shot run uses S+20K labeled examples; a per-class
run uses (S+20)K. Describe the curve as **fit budget at fixed validation support**,
not performance with only S labeled examples available in total.

Emissary Projects currently does not receive this local validation set; report
zero local validation examples consumed and separately record any provider
internal split of the uploaded S examples if exposed. The split is currently
unknown. Do not claim equal total labeled-data consumption between these regimes.
If the research target instead caps *all* labeled data at 5/100, revise this
design before freezing: validation must come out of that cap or tuning must be
fixed without validation. That is a different experiment.

## 4. Dataset, split, seed, and label-description policy

1. Freeze Banking77 source train/test files by content hash and immutable source
   revision. Current CSV URLs point at `master`; row IDs alone are insufficient.
   Use the original test split only for final evaluation, never for fitting,
   hyperparameter selection, description editing, or pilot tuning.
2. Freeze one eligible label pool before running models, using counts alone.
   For the total-shot proposal plus the historical-support reference, require
   at least 100 source-train and 20 source-test examples per class (66 labels in
   the saved source). For 100 fit shots **per class** plus 20 validation, require
   120 source-train examples (49 labels in that source). Recheck against pinned
   bytes. Use strict support: no automatic backoff during final evaluation.
3. Preserve canonical label order from the versioned profile; shuffle it with
   `random.Random(seed)` and take K-prefixes. Within a seed, label sets are nested;
   across seeds, class selection changes. Every method receives identical class
   names/order and the same test IDs within a condition.
4. For each eligible label, start from IDs sorted lexicographically. Shuffle its
   source-train IDs with `random.Random(f"{seed}:protocol-v2:train:{label}")`.
   Reserve the first 20 for validation and keep the remaining ordered sequence
   as fit candidates. Perform this once per seed/label, independent of K and S.
   Do not call the current shared-RNG split separately for each K.
5. Select fit examples using the existing balanced-round-robin selector's rule:
   sorted class names shuffled by `f"{seed}:classes"`, and sorted fit-candidate
   IDs shuffled per label by `f"{seed}:examples:{label}"`. Pass the same seed
   explicitly to every method; the current Emissary selection default is always
   42 and must not silently override repetition seeds. Smaller S is a prefix of
   larger S for the same K. Freeze manifests and reuse them across classifiers.
6. With a fixed **total** budget, per-class fit support decreases as K grows.
   Actual fit sets therefore need not be nested across K. With **per-class**
   budgets, shared classes retain fit IDs across K. Use the 80-per-class reference
   for a comparison that preserves shared-class training support; even there,
   adding classes also adds training examples and changes the learned model.
7. Select 20 test IDs per class once per seed using sorted source IDs and
   `random.Random(f"{seed}:protocol-v2:test:{label}")`. Keep them fixed across
   methods and S, and for shared labels across K. Report the full balanced test
   set and a secondary fixed K=5 test cohort at larger K. Never mix their metrics.
8. Reject overlapping IDs and exact UTF-8 text across fit/validation/test before
   any remote operation, using the existing partition validator. Record a split
   failure; do not silently resample a favorable replacement. Any source cleanup
   must be frozen and hashed before the final manifest.

Use the frozen Banking77 profile
[`canonical_llm_enriched_v1`](../class_definitions_data/banking77/canonical_llm_enriched_v1.json),
SHA-256 `6be474e66645862ffbff383695263b3f9b523bcd216668eebb030d3f0d0bdfc2`.
Its metadata records label-based generation without train/validation/test examples;
its review status is **unreviewed**. Review wording using taxonomy information,
then freeze a new profile/version if edits are needed. No regeneration or edits
in response to model/test outcomes. Subset descriptions exactly for each K.

OpenAI and routing consume those descriptions. Local supervised methods use class
names for targets and do not train on descriptions. Projects SFT consumes the
documented prompt/one-hot completion format; preservation of routing descriptions
is not established. Record the actual information each method receives.

## 5. Model selection and reproducibility

| Method | Frozen proposed settings |
| --- | --- |
| TF-IDF + LR | Word ngrams (1,2), lowercase, min_df=1, no max_features, no sublinear TF; C=(0.1,1,10), max_iter=2000 |
| MiniLM + LR | Frozen `all-MiniLM-L6-v2`; embedding batch 64; C=(0.1,1,10), max_iter=2000 |
| BERT | `bert-base-uncased`; 3 epochs, batch 16, learning rate 2e-5, weight decay .01, max_length=128; minimum validation-loss epoch, first epoch wins an exact tie |
| OpenAI | Pin an available dated snapshot explicitly; reasoning `minimal`; existing structured label-only prompt |
| Emissary routing | Freeze experiment IDs, class configuration and any exposed model/version identifiers |
| Emissary Projects | Provisional `Llama-3.2-1B-Instruct`; 3 epochs, learning rate 2e-4, train/eval batch 2/1; explicit checkpoint/deployment identity; subject to live acceptance |

For both LR methods retain current selection: highest validation accuracy,
then first C in configured order on an exact tie. Save every candidate's score
and selected C. Issue #7 verifies that this explains the five-class underconfidence;
it is not a metric bug. Do not retrospectively substitute a C chosen after
viewing test calibration. A validation-log-loss tie rule or post-hoc calibration
would be a separately preregistered extension, not an unreported change.

The README describes OpenAI as snapshot-pinned, but `config.py` currently sets
`gpt-5-nano`. Resolve that mismatch and verify an immutable model identifier before
the final run. Pin Hugging Face snapshot revisions and dependency versions; save
code commit, hardware, device, thread counts, seeds, prompts, definitions, source
hashes, and selected settings. Seeds do not guarantee deterministic remote APIs;
state what the provider exposes and retain request IDs/resolved model names.

## 6. Metrics, uncertainty, and failed predictions

Primary quality measure: **macro-F1 over the configured K labels**. Report accuracy
alongside it. Primary probability measures: multiclass log loss and Brier; report
top-label ECE and adaptive ECE as calibration summaries. All lower-is-better
probability measures retain existing conventions: 10 bins, natural-log loss with
epsilon=1e-15, and Brier summed over classes then averaged over examples, range
[0,2]. Probability maps require correct labels, normalization and consistency.

No confidence/probabilities means unavailable ECE; confidence alone permits ECE
but not log loss/Brier. Label-only OpenAI output stays unavailable for all four.
Do not derive probabilities from labels or compare missing values as zeros.

For every K/budget/method, publish all seed results, their arithmetic mean and
sample standard deviation. For quality and probability metrics, propose **2,000
paired hierarchical bootstrap replicates**, seed 20260915: resample the five
seed blocks with replacement, then test IDs with replacement within each gold
class of each selected block. Recompute per-run metrics and average across
blocks. Use percentile 2.5/97.5 bounds. Reuse draws across methods/budgets for
paired differences; fixed-cohort comparisons use matched IDs across K.
Intervals are conditional on this dataset/design, not independent-dataset
generalization; overlapping cohorts and five seeds limit uncertainty estimates.
Do not pool all seeds' rows as independent observations or drop failed seeds.

Distinguish these outcomes explicitly:

- **Invalid/missing prediction:** retain its intended test ID, failure stage and
  reason. It contributes an incorrect classification; for macro-F1 it is a false
  negative for its gold label, averaged over the configured labels only. It does
  not create an extra "error" class in the macro average.
- **Unavailable probability output by design:** classification remains eligible;
  probability metrics are unavailable, with reason.
- **Malformed/missing probabilities in a probability-capable run:** record a
  failure and keep primary full-cohort probability metrics unavailable; never
  replace them with a one-hot or silently compute on successful rows only.
- **Training/preparation failure:** report the planned cell as failed, with no
  model quality estimate. Report completed/planned runs and prediction coverage;
  do not average away missing seed blocks. Mark a complete-matrix comparison
  unavailable when a required cell is missing.
- **Predeclared infeasible condition:** report separately from execution failures.

The current runner fails a run on an invalid output and does not persist every
partial prediction. Failure-aware artifacts/aggregation are therefore a concrete
implementation dependency. Proposed final inference policy is one application
attempt, no automatic SDK retries, with explicit timeouts saved in the manifest.
Any operational rerun preserves the failed attempt and all its incurred cost.
Never automatically retry ambiguous training/deployment submissions.

## 7. Latency and cost

Primary latency summary: per-example p50 and p99; include mean and per-seed
values. Use sequential single-example inference, fixed hardware/network context,
and record client timeouts/retries. Exclude training and loading from inference
latency but report their wall times separately. Use five source-train inputs for
warmup per prepared model, outside the scored cohort; count their cost. Schedule
method/condition order with a saved seeded permutation to limit ordering bias.
Do not combine GPU and CPU runs into one timing estimate. Remote timings include
client-observed network/service time; existing local adapter timings have different
boundaries, so retain them as adapter timing and add a common inference boundary
before presenting a direct comparison. With 100 examples at K=5, p99 is descriptive
and unstable. Do not claim precise tail latency from the tiny pilot.

Report measured/estimated status and pricing provenance for: preparation/training,
deployment idle time, warmups, inference, retries and failed calls. Report USD per
1,000 attempted inputs as an operational measure, and successful-input coverage;
label the existing reducer's per-record denominator separately when appropriate.
Local API charges are zero but compute cost is unknown unless measured/priced.
Unavailable provider charges remain unavailable, not zero.

Freeze the owner-approved total and pilot caps before any paid work. Maintain a
ledger of incurred plus conservatively committed charges; do not submit an
operation if its bounded maximum would exceed the remaining allocation. Pilot
spend counts against the total. If the provider cannot bound a paid operation,
that arm is blocked under a hard-dollar-cap protocol until a provider/account
limit or other enforceable bound exists. Emissary's max-new-job flag is a job
count, not a USD spending cap. The current runner lacks complete cost attribution
and campaign USD enforcement; reducers alone do not satisfy this requirement.

## 8. Pilot and release dependencies

Technical smoke: AG News/local fixtures, then Banking77 K=5, seed 42. Use a
source-train-only split for pilot development and validation; reserve the original
test for the frozen final campaign. Check both 5/100 settings under the confirmed
unit with two pilot-held-out examples per class. Establish feasibility, class
coverage, saved artifacts, pricing and timings; do not select methods from pilot
quality or report pilot metrics as final results. Do not repeat successful
historical uploads without checking continuation hashes/IDs.

Freeze the final manifest only after the pilot, with all requested cells,
infeasible cells and exclusions listed before evaluation. If estimated spend
does not fit the cap, revise scope explicitly before final inference.

| ID | Dependency / decision | Required resolution |
| --- | --- | --- |
| D1 | Owner: shot unit and interpretation of total labeled resources | Confirm total vs per-class, and accept fixed extra validation support or revise budgets |
| D2 | Owner: USD limits | Replace proposed USD 50/5 with confirmed total/pilot caps |
| D3 | Owner: final dataset scope | Confirm Banking77-only; any second dataset needs source/revision, license, labels, support, frozen definitions and its own K grid |
| D4 | Dataset freeze | Pin source bytes; freeze eligible labels, exact splits and duplicate policy |
| D5 | Sampling implementation | Stable per-label validation, explicit shared fit-budget selection, paired test IDs and recorded seeds |
| D6 | Issue #7 integration | Merge audit validation/metadata; retain documented C tie rule |
| D7 | Model/description freeze | Review profile, pin model revisions and resolve README/config snapshot mismatch |
| D8 | Emissary live acceptance | Recheck account readiness, pricing/bounds, checkpoint policy and output contract; SFT and routing are different products/base models |
| D9 | Metrics/reporting | Failure artifacts, configured-label macro-F1 aggregation, paired bootstrap and common timing boundaries |
| D10 | Cost execution controls | Complete charge ledger, resolved price sources and enforceable caps; record unavailable components |

Completion of issue #6 means the protocol and owner decisions are recorded, with
remaining engineering/API dependencies explicit. It does not require implementing
all those dependencies or executing the final benchmark in this documentation change.
