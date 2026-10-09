# Perplexity Decisions

`PerplexityDecisionsClassifier` adds a zero-shot method with a full native
probability distribution. It uses `POST https://api.perplexity.ai/v1/decisions`,
`state` for unchanged input text, and one named `choice` question whose `criteria`
map frozen class names to descriptions. The default model is `pplx-decider-v1.1-27b`.
It consumes no training, validation or in-context labels.

Set `PERPLEXITY_API_KEY` in your local `.env`. Any standard Perplexity API key
works; create one at [the Perplexity console](https://console.perplexity.ai).
The committed `.env.example` and local `.env` have a `replace_me` placeholder;
replace it with your own key. No separate Decisions key is required.
Optional settings are `PERPLEXITY_DECISIONS_MODEL`, `PERPLEXITY_DECISIONS_TIMEOUT_S`
(default 30 seconds) and `PERPLEXITY_BASE_URL` (default `https://api.perplexity.ai/v1`).
CLI `--perplexity-model` and `--perplexity-timeout-s` override environment settings.
Existing `requests` and `python-dotenv` dependencies are sufficient.

## Replay the completed study

Run from the repository root:

```bash
# Offline audit and plan; makes no provider calls and needs no API key.
./scripts/run_perplexity_decisions.sh

# Execute the four paid zero-shot cells after configuring the key.
./scripts/run_perplexity_decisions.sh --execute
```

The replay uses K=5/10/15/20, seed 42, 40 test examples per class: 2,000
predictions. It reuses the existing GPT-6 replay's reference audit to verify the
frozen CSV hashes, definitions, test IDs, order, text, gold labels and comparable
historical predictions before any API request. Source data is read locally from
`artifacts/v2_source`; reference files are `reports/v2/manifest.json` and
`reports/v2/results.json`. `--source-dir`, `--reference-manifest` and
`--reference-results` override those paths.

Each invocation creates a campaign under `artifacts/perplexity_decisions/` with
`campaign.json`, `audit.json`, derived definitions, summary JSON/CSV, and normal
runner directories containing predictions, usage, metrics and cost reports.
`--output-root` overrides the root. The script stops on its first failure and
retains earlier predictions and all observed usage, including malformed replies.
There is no automatic retry or resume; rerunning `--execute` creates a new
campaign and repeats its requests. Completed publication reports are updated
only after actual evidence exists.

## Use the general benchmark

Both Banking77 campaign entry points accept `--classifiers perplexity-decisions`.
For example, to plan a smaller campaign with the maintained v2 runner:

```bash
PYTHONPATH=src venv/bin/python scripts/run_banking77_scaling_benchmark_v2.py \
  --classifiers perplexity-decisions --class-counts 5 --seeds 42 \
  --matched-budgets 0 --budget-unit per_class --warmup-examples 0 \
  --pricing pricing/perplexity_decisions_2026-10-09.json --dry-run
```

Remove `--dry-run` to execute that campaign. Nonzero matched fit or validation
budgets are recorded as unsupported for this zero-shot adapter.

## Measurement and billing

The benchmark's `confidence` is the probability of the predicted class; the
API's separate certainty estimate is stored as `raw_response.provider_confidence`.
Probability vectors retain the provider values and existing sum tolerance of
0.0001. Unknown/missing labels, nonfinite values, invalid sums or a choice below
the maximum probability fail validation. Exact ties preserve the provider choice.

Inference sends one input per request, sequentially, with no automatic retries
or redirects. HTTP 429 and server errors stop the run; there is no hidden backoff
inside measured inference. Request IDs, request hashes, model IDs, token usage
and failed attempts are recorded without credentials or provider error bodies.
The documented organization rate limit is 10 requests per second.

The dated rate card prices both documented models at $0.02 per million reported
`usage.input_tokens`; output tokens are free and there is no per-request fee.
It reuses the existing `all_input_tokens` accounting. Estimates exclude taxes,
discounts and custom terms. Custom endpoints, unlisted resolved models and missing
usage remain unavailable. `--pricing` selects an alternative rate card.

The API permits up to 255 choice options, input below 262,144 tokens, and a
32 MiB body. The adapter checks label count and body size, and conservatively
estimates text tokens using UTF-8 payload bytes plus a 1,024 framing allowance.
It never truncates inputs or class definitions.

Contract, supported models, limits and pricing verified on 2026-10-09 against
the [official Decisions quickstart](https://docs.perplexity.ai/docs/decisions/quickstart).

## Offline validation

```bash
PYTHONPATH=src venv/bin/python -m pytest -q \
  tests/test_perplexity_decisions.py tests/test_perplexity_decisions_campaign.py
```

Tests use simulated HTTP responses and temporary datasets; no external API
requests are needed.
