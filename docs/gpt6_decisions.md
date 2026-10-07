# GPT-6 Decisions on the completed Banking77 matrix

This extension adds OpenAI Decisions through the existing `Classifier` lifecycle,
`Prediction`, runner, measurement ledger and metrics. It preserves the GPT-5
generative adapter and historical study results. The public report and brief now
include an offline supplement from the user's completed campaign.

## Scope

The reference is the completed 64-condition study recorded in
`reports/v2/manifest.json` and `reports/v2/results.json`. OpenAI generative, Jev
and Emissary routing each completed four zero-shot conditions. Local models used
20/50/100 fit examples/class; Qwen and Llama SFT used 20/100. Those supervised
budgets share the same test cohorts and do not imply additional Decisions fits.

| K | Seed | Fit/class | Validation labels consumed | Test/class | Predictions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 42 | 0 | 0 | 40 | 200 |
| 10 | 42 | 0 | 0 | 40 | 400 |
| 15 | 42 | 0 | 0 | 40 | 600 |
| 20 | 42 | 0 | 0 | 40 | 800 |

The four cells total 2,000 predictions, with one choice question per input,
sequential requests, runner batch size one, no warmup and no automatic retries.
The script reconstructs the same 125 source-training candidates per class and
20% reserved partition using existing sampling helpers. The runner selects a
zero-label matched pool. Decisions consumes no fit or validation labels.

Older smoke probes and retired matrices are not added to this experiment.
The test has already been observed in the previous study; this extension adds
a model to that study rather than establishing a new unseen test.

## Configuration and execution

Use the repository's existing `venv` and dependencies. The adapter uses
`requests`, following the Jev transport pattern, because the current `openai<3`
dependency does not provide Decisions. No SDK upgrade is needed.

Set `OPENAI_API_KEY` in your shell or the gitignored `.env`. Optional environment
variables are documented in `.env.example`:

| Variable | Default / meaning |
| --- | --- |
| `OPENAI_DECISIONS_MODEL` | `gpt-6-luna`; independent of the generative baseline |
| `OPENAI_DECISIONS_TIMEOUT_S` | `120` seconds per request |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1`; HTTPS API base, without `/decisions` |
| `OPENAI_ORG_ID` | Optional `OpenAI-Organization` header |
| `OPENAI_PROJECT_ID` | Optional `OpenAI-Project` header |

From the repository root, execute **all four conditions** with:

```bash
bash scripts/run_gpt6_decisions.sh --execute
```

Omit `--execute` for an offline audit and runner plan. It requires no API key
and constructs no HTTP session. For example:

```bash
bash scripts/run_gpt6_decisions.sh --output-root /tmp/gpt6-decisions-plan
```

Both modes require the existing local frozen inputs: `artifacts/v2_source/`,
the reviewed definitions, the reference manifest/results, and their raw
prediction evidence. Before inference, every completed comparable condition is
checked for the exact test IDs, order, texts and gold labels. Frozen source
and definitions are verified by SHA-256. Missing or changed evidence fails
before any paid request; the script does not substitute freshly downloaded data.

Available overrides include `--decisions-model`, `--decisions-timeout-s`,
`--pricing`, `--source-dir`, `--reference-manifest`, `--reference-results` and
`--output-root`. These are recorded in campaign/run artifacts. Each invocation
creates a new timestamped campaign unless `--resume CAMPAIGN_DIRECTORY` is supplied.
Refusals are recorded and execution continues with the next input. Other errors
stop execution, retaining partial evidence, without automatic retries.

To continue an existing campaign after a refusal or HTTP 503, preserving its completed and
partial runs, use:

```bash
bash scripts/run_gpt6_decisions.sh --execute \
  --resume artifacts/benchmark_runs/20261007T182643602204Z
```

Omit `--execute` to inspect the remaining request count without API calls. Resume
checks all existing cells before sending requests, validates source/configuration,
definitions, model, transport, pricing, sample IDs, texts, labels, predictions and
usage, and skips both successful predictions and recorded refusals. It accepts
the earlier plain `ValueError` refusal when corroborated by the recorded HTTP-200
failed attempt. A recorded HTTP 503 is also resumable when the status and usage
ledger agree: the failed input remains pending and is retried first in a new
segment, while earlier successful predictions and refusals are skipped. The old
segment, failed attempt, and cost report remain intact. Missing usage on a 503
remains unknown, rather than being treated as zero cost. Wait before resuming a
503; a subsequent 503 still stops execution without automatic retries.
Ambiguous/in-progress runs and other failures require inspection.
A campaign lock prevents simultaneous executions. Repeating a completed resume
makes no calls; repeating `--execute` **without** `--resume` starts a fresh campaign.

## Adapter and artifacts

The request uses `POST /v1/decisions`, shared `input` text, and a named
`classification` question with `type: choice`. Each choice carries the existing
class name and frozen description. No gold label, training example, reasoning
setting, generative JSON schema or self-reported probability prompt is sent.

The returned `probabilities` array is mapped by its string `value` into the
benchmark's label-to-probability mapping, in configured class order. Every label
must occur exactly once. Values must be finite and in [0,1], sum to one within
the shared `1e-4` tolerance, and agree with the chosen maximum-probability label.
Values are preserved without renormalization. Exact ties keep the provider's choice.

Benchmark confidence is the probability of the predicted label, as in Jev.
The API's separate native confidence is retained in `raw_response`. Refusals,
missing or malformed distributions and transport errors fail the run while
preserving usage and previously validated predictions. Requested/resolved model,
request ID, payload digest and transport configuration remain recorded.

Outputs use `artifacts/benchmark_runs/<timestamp>/`, existing run names, frozen
subset definitions, `campaign.json`, `summary.json` and `summary.csv`. Each run
uses the normal `config.json`, `status.json`, `fit_metadata.json`,
`predictions.jsonl`, timing/usage ledgers, `metrics.json`, `operational_report.json`
and `cost_report.json`. These operational JSON artifacts are normal runner outputs.

## Published results and offline reproduction

Campaign `20261007T182643602204Z` completed all 2,000 inputs with 1,990 classifications
and 10 refusals (0/2/5/3 at K=5/10/15/20). K=5 has full coverage and 93% accuracy;
the other cells have unavailable full-cohort quality metrics. K=20 includes one
HTTP 503 and its explicit retry, with unknown usage on the failed attempt.
The earlier abandoned campaign and any future refusal retries are excluded.

`reports/v2/decisions_results.json` retains the audited supplement, including source
hashes, coverage, quality, paired K=5 intervals, cost estimates and segment timings.
The original `reports/v2/results.json` remains unchanged. The report, public brief,
combined CSV and figure render from both versioned summaries without credentials:

```bash
PYTHONPATH=src venv/bin/python scripts/build_v2_reports.py
```

To audit and recompute the Decisions supplement from the local evidence:

```bash
PYTHONPATH=src venv/bin/python scripts/build_v2_reports.py \
  --decisions-root artifacts/benchmark_runs/20261007T182643602204Z
```

Both commands are offline. Raw evidence stays under ignored `artifacts/`.

### Diagnosing a refusal

`OpenAIDecisionsRefusalError` means the HTTP request succeeded but the Decisions
answer had `type: refusal`, with no class label or distribution. The error includes
the sample ID and request ID; `usage.jsonl` retains `decision_answer_type`, HTTP
status and billing units. This differs from HTTP 429 rate limits or HTTP 503
overload errors. The provider does not expose the refusal's cause in this answer.

The runner preserves earlier predictions and marks that execution segment failed.
The campaign catches only this explicit refusal, records it, and uses the same
runner for the remaining test inputs in a new `__partNNNN` directory. Each source
run remains immutable. Inputs, class descriptions and prompts remain identical;
the refused sample is not retried or replaced with a fabricated prediction.

Each cell also has `cells/<cell_id>/coverage.json`, `outcomes.jsonl` (every attempted
input, in original order, with `classified` or `refused` status), `refusals.jsonl`,
and `predictions.jsonl` (only actual predictions, preserving source-run provenance).
`coverage.json` records source-file hashes, successful/refused/pending counts and
the complete planned IDs. A cell that attempted all inputs but has refusals is
`completed_with_refusals`, not an ordinary completed classification result.

For those cells, full-cohort quality and latency metrics remain unavailable in
the cell summary; raw per-segment timing remains available. The generic artifact
evaluator rejects a `predictions.jsonl` whose sibling coverage manifest declares
missing predictions, preventing accidental scoring of only the accepted subset.
Cost totals sum the standard segment ledgers, including all refused attempts;
cost per 1,000 successful classifications uses the successful count. No refusal
is silently omitted from coverage or billing. A later analysis can explicitly
choose an abstention scoring policy using the complete outcomes.

See the [Decisions refusal schema](https://developers.openai.com/api/reference/resources/decisions/methods/create)
and [OpenAI HTTP errors](https://developers.openai.com/api/docs/guides/error-codes).

Expected metrics are accuracy, macro-F1, fixed/adaptive ECE, multiclass log loss,
Brier score, mean/p50/p95 latency, throughput and inference cost. Existing p99
minimum-observation rules apply: none of these individual cells has 1,000 examples.

The dated `pricing/decisions_2026-10-07.json` uses the published Decisions-specific
base rate of $0.10 per million reported input tokens, with no additional cache
read/write or output charges. It reuses the existing endpoint-specific
`all_input_tokens` billing path. Usage stays in its native API shape. Cost is an
estimate for short text without regional premiums or custom terms; use `--pricing`
for other account terms. Unlisted resolved models, custom endpoints or missing
usage remain unavailable rather than being priced as the generative baseline.

OpenAI currently documents `gpt-6-luna` as the supported Decisions model in public
beta. Its alias is recorded along with the resolved response model; no dated
snapshot or account access is assumed. No live request is necessary for the tests.

Sources checked on 2026-10-07:
[Decisions guide](https://developers.openai.com/api/docs/guides/decisions),
[HTTP schema](https://developers.openai.com/api/reference/resources/decisions/methods/create).

## Offline validation

```bash
PYTHONPATH=src venv/bin/python -m pytest -q tests/test_openai_decisions.py \
  tests/test_gpt6_decisions_campaign.py tests/test_jev.py \
  tests/classifiers/test_openai.py tests/test_costs.py tests/test_runner.py
```

These tests use simulated HTTP responses and temporary datasets, including
probability normalization, ties, refusals, failed-call usage, endpoint pricing,
configuration, exact cohort replay and both existing campaign entry points.
The generic scripts also accept `--classifiers openai-decisions`; use the
dedicated replay command above to reproduce the completed study exactly.
