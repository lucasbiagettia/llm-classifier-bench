# Matched-budget implementation validation

Validated on 2026-09-21 at code commit
`4f5b2f0a329c47930f4ec5dffdf6f3a75832c42a`, with a clean working tree.

```bash
PYTHONPATH=src venv/bin/pytest -q -m 'not integration'
PYTHONPATH=src venv/bin/python scripts/probe_matched_budgets.py
```

Result: **246 passed, 3 external integration tests deselected**. The offline smoke
passed every assertion, with zero paid requests and zero model downloads.

[Saved validation summary](../artifacts/matched_budget_validation/issue14/validation.json)
links all eight runs. Each budget pairs real local TF-IDF + LR with mocked OpenAI
in-context requests, using the same pool hash and exact sample IDs. The two-class
fixture has two separate validation examples available in every cell.

| Total shared examples | TF-IDF | Mock OpenAI | Class coverage |
| --- | --- | --- | --- |
| 0 | Unsupported | Completed, zero demonstrations | 0/2 |
| 1 | Unsupported | Completed, one demonstration | 1/2 |
| 2 | Completed | Completed, two demonstrations | 2/2 |
| 5 | Completed | Completed, five demonstrations | 2/2 |

At budget 5, TF-IDF consumes 5 training + 2 validation labels; OpenAI consumes
5 context + 0 validation labels. The distinct totals are 7 and 5, explicitly
reported rather than presented as equal total label consumption. Smaller shared
budgets are nested prefixes. Recorded mock requests include every demonstration
and exclude reserved validation examples.

The test suite additionally exercises real LR fitting over deterministic fixture
embeddings, both campaign CLIs, plans for all five adapters, repetition seeds,
Emissary pool identity and job bounds, insufficient source support, context
preflight before any call, and provider context rejection with retained usage
and cost artifacts. Existing full-training behavior remains covered.

This is implementation evidence. OpenAI responses and usage are simulated;
MiniLM fixture embeddings are not downloaded model outputs. BERT and Emissary
planning tests do not establish live training acceptance. Fixture quality and
latency are not final benchmark findings. The final paid evaluation, provider
readiness, context settings and owner budget decisions remain separate.
