# Add GPT-6 Decisions zero-shot classification to the completed Banking77 benchmark

## Motivation

Evaluate OpenAI's GPT-6 Decisions classification/choice primitive alongside the benchmark's completed Banking77 baselines, including GPT-5 generative zero-shot, Jev, Emissary routing, local supervised models, and Qwen/Llama SFT. Native class probabilities allow the existing calibration metrics to be compared as well as classification quality.

This is an additive extension on branch `feat/gpt6-decisions`. Preserve the existing benchmark design, splits, frozen descriptions, metrics, raw artifacts, naming conventions, and prior results.

## Implementation scope

- Add `OpenAIDecisionsClassifier` implementing the existing `prepare -> fit -> predict` interface and returning ordinary `Prediction` objects.
- Call `POST /v1/decisions` with one named `choice` question per input. Reuse existing class names and descriptions; send no target-bearing metadata or training examples.
- Convert the API's probability array into the existing label-to-probability mapping. Validate exact label coverage, uniqueness, finite values, the shared probability-sum tolerance, and consistency with the selected label. Preserve exact ties and rounded values.
- Use the chosen-label probability as benchmark confidence. Retain native confidence separately, plus requested/resolved model identifiers, request ID, and payload digest.
- Preserve failure evidence and API usage, including refusals and malformed responses. Use sequential requests with explicit timeout and no automatic retries.
- Configure credentials through `OPENAI_API_KEY`, with optional `OPENAI_DECISIONS_MODEL`, `OPENAI_DECISIONS_TIMEOUT_S`, `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, and `OPENAI_PROJECT_ID`. Never commit credentials.
- Integrate `openai-decisions` into both existing Banking77 campaign entry points.
- Reuse the runner's usage ledger and endpoint-specific input-token cost path. Add a separate dated Decisions rate card; do not reuse generative model pricing.
- Add focused offline tests and a single command for the exact completed-study replay.

Use the documented HTTP transport, following the existing Jev adapter, to preserve the repository's current OpenAI SDK dependency and generative adapter. Official documentation currently identifies `gpt-6-luna` as the Decisions model; preserve requested and returned identifiers without inventing a dated snapshot.

## Experiment matrix

Reference: `reports/v2/manifest.json` and `reports/v2/results.json`, which record 64 completed conditions across 16 method/budget variants.

Decisions matches the completed zero-shot OpenAI and Jev matrix:

| K / labels | Seed | Fit examples/class | Consumed validation labels | Test examples/class | Total predictions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 42 | 0 | 0 | 40 | 200 |
| 10 | 42 | 0 | 0 | 40 | 400 |
| 15 | 42 | 0 | 0 | 40 | 600 |
| 20 | 42 | 0 | 0 | 40 | 800 |

Four runs, 2,000 predictions, one request/question per text, batch size one, concurrency one, zero warmup, zero automatic retries. Preserve the exact existing label order, frozen reviewed definitions, test IDs/texts/gold labels, source revision/hashes, and sampling/split helpers.

Local supervised budgets (20/50/100 per class) and Qwen/Llama budgets (20/100) share these test cohorts. They are comparison references, not additional supervised Decisions conditions. Nonzero matched-label budgets are explicitly unsupported by this zero-shot adapter. Retired matrices and smoke probes are outside this extension.

## Execution and outputs

Set `OPENAI_API_KEY` in the shell or ignored `.env`, then run from the repository root:

```bash
bash scripts/run_gpt6_decisions.sh --execute
```

Without `--execute`, the command audits the frozen local evidence and writes offline runner plans without API calls. It requires the existing local `artifacts/v2_source` files and raw prediction evidence referenced by the report. It verifies every completed comparable cohort before paid inference.

Outputs remain under `artifacts/benchmark_runs/<timestamp>/`: existing run directories and raw prediction/config/status/fit/usage/timing/metrics artifacts, plus campaign and CSV/JSON summaries. Refusals are recorded separately and execution continues through immutable runner segments. Other failures stop execution without automatic retries. `--resume CAMPAIGN_DIRECTORY` validates and skips previous predictions and refusals, and accepts a corroborated HTTP 503 failure for explicit retry; without it, execution creates a fresh campaign.

Cell-level `outcomes.jsonl` and `coverage.json` retain all attempted inputs and explicit refusal counts. Full-cohort quality metrics remain unavailable if any example was refused; the evaluator must not silently score only accepted examples. Cost totals include refused calls. Resume must preserve original segment artifacts byte-for-byte and reject changed or ambiguous evidence before paid calls.

## Expected metrics and accounting

- Accuracy and macro-F1.
- Top-label ECE, adaptive ECE, multiclass log loss, and multiclass Brier score.
- Mean/p50/p95 latency and throughput using the existing runner boundary.
- P99 only when the existing minimum-observation requirement is satisfied; individual cells here remain below it.
- Input-token inference cost and cost per 1,000 successful predictions, including billed failed attempts.

The dated Decisions rate card uses the published endpoint-specific base price of $0.10 per million reported input tokens with no additional cache-read, cache-write, or output charges. The estimate assumes short text and no regional premiums/custom terms. Preserve unavailable costs for missing usage, unlisted resolved models, or custom endpoints.

Official sources:
- [Decisions guide and endpoint pricing](https://developers.openai.com/api/docs/guides/decisions)
- [Decisions HTTP request/response schema](https://developers.openai.com/api/reference/resources/decisions/methods/create)

## Acceptance criteria

- [ ] Implementation lives on `feat/gpt6-decisions`.
- [ ] Existing adapters, default classifier selection, sampling, splits, metrics and historical outputs retain their behavior.
- [ ] Both maintained campaign entry points accept `openai-decisions`.
- [ ] Native full probabilities reach the normal calibration metrics without model-specific analysis code.
- [ ] Refusals, malformed output, and transport/HTTP failures retain prior predictions and usage; no hidden retries.
- [ ] A refusal does not abort remaining samples/cells. Explicit resume skips successful and refused samples, and full-cohort metrics never hide refusals.
- [ ] Credentials and optional provider configuration come from environment variables or explicit CLI overrides.
- [ ] Offline tests cover output normalization, ties, native confidence separation, failures, usage/cost replay, cohort auditing, dry runs, and campaign integration.
- [ ] Offline audit verifies all completed Banking77 reference cohorts and plans the four exact zero-shot cells.
- [ ] One command executes the whole matrix and writes ordinary raw benchmark results.
- [ ] Document the zero-shot scope, alias/account-access assumptions, dated pricing, and fresh-campaign behavior.
- [ ] Do not run paid/full experiments during implementation.
- [ ] Publish the user-run campaign in the public report, README and public brief, retaining the original 64 condition objects and explicitly reporting refusals and unknown costs.
- [ ] Version an audited Decisions summary that supports offline clean-clone rendering; leave raw evidence outside Git.

Implementation instructions and assumptions: `docs/gpt6_decisions.md`.
