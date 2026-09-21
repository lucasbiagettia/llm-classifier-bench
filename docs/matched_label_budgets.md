# Matched labeled-example budgets — issue #14

Both maintained Banking77 campaign scripts support a separate matched-budget
comparison. A budget is a pool of **distinct labeled examples**, supplied as
training data to supervised methods or as demonstrations to OpenAI. It is not
an epoch count or the cumulative number of tokens billed across predictions.

## Running and planning

CPU-only example (Banking77 loading may download the source dataset):

```bash
PYTHONPATH=src python scripts/run_banking77_scaling_benchmark_v2.py \
  --classifiers tfidf --class-counts 5 10 20 25 --seeds 42 43 \
  --matched-budgets 0 5 100 --budget-unit total --validation-budget 0 \
  --train-per-class 100 --test-per-class 20 --strict-support \
  --output-root artifacts/matched_budgets
```

Add `--dry-run` to save every planned cell and its IDs without fitting, loading
model weights, constructing provider clients or calling APIs. Add
`sentence-transformer bert openai emissary` to `--classifiers` to plan all methods.
Source-data loading still takes place.

For matched OpenAI runs, supply `--openai-context-window-tokens` explicitly from
the chosen model's documented context limit. An unspecified limit produces an
`unsupported` cell. `--openai-completion-reserve-tokens` defaults to 4096 and is
also sent as `max_completion_tokens`; `--openai-framing-allowance-tokens` defaults
to 1024. These are operator settings, not discovered provider capabilities.

For nonzero Emissary budgets, retain the existing explicit
`--emissary-mechanism project_fine_tuning`, project and base-model settings.
Live execution still requires its existing unpriced-training acknowledgment and
job bound. The bound covers all budgets × seeds × class counts. Continuation IDs
must identify exactly one nonzero budget/seed/class-count condition. Missing
mechanism, incomplete class coverage and unmatched selections are unsupported.
A dry-run does not establish account/payment readiness or provider acceptance.

Do not combine `--matched-budgets` with Emissary-only shot/unit flags. Shared
selection uses the campaign repetition seed, including for Emissary; its old
independent default seed cannot change the pool. Zero-shot controls occur once
per requested K/seed; budgets are deduplicated.

Omit `--matched-budgets` to run the existing **full-training reference** mode.
It keeps existing adapter-specific behavior (including zero-shot OpenAI and
Emissary-only budgets). Its manifest, config and summary explicitly say
`full_training_reference`. It never shares a campaign summary with matched runs.
Do not pool the two regimes in downstream plots.

## Pools, coverage and validation

1. Keep the original held-out test split. Derive fit candidates and a reserved
   validation partition using the existing deterministic stratified split.
   Validate IDs and exact text across the entire partitions before subsampling.
2. Select the fit/context pool by `balanced_round_robin_v1`: sort IDs within
   classes, shuffle with the recorded seed per label, then interleave the seeded
   class order. Each of K classes receives `N // K` examples, with one extra for
   the first `N % K` labels. Insufficient assigned support makes the cell
   unsupported; examples are never duplicated or redistributed.
3. `--budget-unit total` means N examples altogether. `per_class` means N×K.
   Smaller budgets are prefixes of larger ones for the **same K/seed/split**.
   Selection is independent of classifier and input ordering. The current
   fit/validation splitter can change shared-label membership across K; no
   cross-K fit identity guarantee is made.
4. Select `--validation-budget V` **total additional** examples from the reserved
   validation partition with the same balanced policy. Default V=0 makes the
   comparison usable without extra labeled selection data. Unused reserved
   examples are withheld, never moved back into the fit pool. V>0 needs a positive
   `--validation-fraction` and sufficient per-label support.
5. TF-IDF, MiniLM and BERT consume that separate validation set for selection.
   OpenAI and Emissary ignore it and report zero consumed validation examples.
   Therefore V>0 is a fit/context-budget comparison at fixed additional validation
   availability, not equal total label consumption across all methods. To reserve
   20 consumed validation examples/class, run each K with V=20×K explicitly.

At V=0, existing training rules are preserved: TF-IDF uses `fallback_c`, MiniLM
retains the first configured C (its existing grid still runs), and BERT retains
the last epoch. At V>0, the LR methods select validation accuracy, with first-C
wins ties; BERT selects validation loss. There is no added final refit.

## Supported matrix

| Method | Budget 0 | 0 < total < K | Full coverage | How labels are used |
| --- | --- | --- | --- | --- |
| TF-IDF + LR | Unsupported | Unsupported | Supported with usable vocabulary | Vectorizer and LR fitting; separate validation selection |
| MiniLM + LR | Unsupported | Unsupported | Supported | Frozen embeddings, LR fitting; separate validation selection |
| BERT | Unsupported | Unsupported | Supported | Parameter training; separate validation epoch selection |
| OpenAI | Zero-shot | Supported | Supported | All selected examples, in order, in every prediction prompt; no parameter fitting |
| Emissary routing | Supported | Unsupported | Unsupported for nonzero budgets | Class definitions only |
| Emissary Projects SFT | No SFT at zero | Unsupported | Conditional on explicit configuration and provider readiness | Same selected IDs uploaded for parameter training |

OpenAI cells additionally require context preflight to pass. All methods need
sufficient source support. `5 total` at K=5 has full coverage; at K=10/20/25,
OpenAI remains eligible while supervised methods are explicitly unsupported.
`100 total` covers all four proposed K values if source support is sufficient.
`5/100 per_class` has full coverage but requires 5/100 fit examples per label
*after* the split; the ordinary `--train-per-class 100` with 20% validation cannot
supply 100 fit examples per class.

Matched labeled data does not equal matched pretrained information, model
capacity, mechanisms or compute. In-context labels are repeated in every API
request and count toward inference tokens/cost; distinct labeled exposure is
counted once. Projects SFT uses a separate base model, not a trained routing
experiment. Emissary may reorder the same selected pool internally; its actual
upload order/hash is recorded in its existing preparation/fit metadata.

## Context handling

The preflight checks **all test inputs and configured warmup inputs before the
first request**. It serializes the actual messages and response schema, counts
UTF-8 bytes as a conservative byte-BPE token proxy, adds the explicit framing
allowance and completion reserve, and compares with the declared context window.
This is not a tokenizer/provider measurement. It may reject prompts that would
fit; provider-specific framing can differ from the allowance. No model context
limit is guessed, and no examples or text are silently truncated to make a cell fit.

A provider `context_length_exceeded` response also becomes `unsupported`, retaining
failed-attempt usage/cost artifacts. Other provider errors remain execution
failures. Actual billed tokens continue to come from provider usage responses,
never from the preflight proxy. OpenAI remains label-only: calibration metrics
stay unavailable.

## Artifacts and verification

Each run retains:

- `config.json`: comparison regime, requested budgets, exact fit/validation/test
  IDs, frozen definitions and model/settings. Impossible source budgets retain
  the requested configuration with empty selected-ID lists.
- `labeled_budget.json`: ordered selected IDs/text/labels/content hashes, seed,
  per-class counts, coverage, separate validation selection, and `pool_sha256`.
  Same hash identifies the same fit/context **and validation** pools across methods.
- After fit, `consumption`: actual `training_examples_used`,
  `context_examples_used`, `validation_examples_used`, and their distinct-label
  sum `total_labeled_examples_used`. Planned/unsupported-before-fit cells have
  no measured consumption; summaries show null rather than invented usage.
- OpenAI `context_plan.json`: per-request prompt hash and estimated context
  requirement. `fit_metadata.json` records demonstration IDs/order and policy.
- Existing prediction, preparation, latency and monetary-cost artifacts.
- `status.json` and summary rows distinguish `completed`, `dry_run`,
  `unsupported` (with reason), and `failed`. Unsupported cells carry no quality
  metrics and must not be silently omitted when reporting matrix coverage.

Offline tests use real sklearn training, deterministic fixture embeddings,
recorded mock OpenAI requests, and Emissary preparation plans. They exercise both
campaign entry points, nested/shared pools, validation exposure, partial coverage,
context preflight/provider rejection, source shortages and whole-campaign job
bounds. The saved smoke in `artifacts/matched_budget_validation/issue14` is
reproduced by `scripts/probe_matched_budgets.py`; it is implementation evidence,
not comparative model-quality or provider-performance evidence. See the
[validation note](matched_budget_validation.md) for recorded results.
