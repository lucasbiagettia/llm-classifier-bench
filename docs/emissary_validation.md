# Issue #13 validation evidence

Validated 2026-09-07 America/Mexico_City with the existing Python 3.14.7
environment; no dependency upgrades.

## Automated tests

- `PYTHONPATH=src pytest -m "not integration" -q`: **118 passed, 3 deselected**.
- Focused Emissary and campaign suite: **59 passed**.
- `python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed.

The Projects fine-tuning tests exercise the actual low-level client, classifier,
runner, artifact writer and metrics code against ordered mocked HTTP responses.
They verify:

- exact documented JSONL label-map serialization and the 100 MB upload limit;
- exact multipart dataset upload, training-job parameters, deployment payload and
  held-out classification requests;
- project/model validation before mutation;
- dataset profiling, training and deployment polling;
- success, terminal failure and bounded timeout behavior;
- no retry after an ambiguous training submission;
- immutable checkpoint selection and response model/class-space validation;
- milestone persistence of dataset, job and deployment IDs;
- continuation without repeating non-idempotent submissions, guarded by the
  original JSONL SHA-256 and parent resource IDs;
- signed download URL redaction and unavailable-cost reporting.

The full runner fixture uploads four selected training examples, observes a job
transition from Running to Success, chooses checkpoint 2, observes a Deployed
deployment, classifies two held-out examples and verifies the persisted metadata
and metrics.

These are offline contract tests. They establish request construction and adapter
behavior; they do not claim that a paid provider training job completed.

## Real Banking77 dry run

The cached Banking77 v2 campaign completed with **3 planned, 0 failed** for
0/5/100 total-shot conditions, using 112 fit, 28 validation and 2 identical test
examples across conditions. Selected classes were `beneficiary_not_allowed` and
`wrong_amount_of_cash_received`.

- 5 shots selected 3/2 examples across the two classes; its exact provider JSONL
  was 765 bytes with SHA-256
  `6752a4962bdc4c437f89efd6b4108188edd38b242e8ebf7a3e428a0ea2fa47f4`.
- 100 shots selected 50/50 examples.
- The 5-example selection was the ordered prefix of the 100-example selection.
- Both nonzero plans list project/model validation, dataset upload/profiling,
  training, checkpoint, deployment and held-out classification operations.
- No Emissary client was constructed and no provider operation was invoked.

The local evidence is outside the repository at
`/tmp/emissary_project_sft_validation/20260907T161755098667Z/`.

The command used a local Arrow-cache loader and then invoked:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src:scripts \
  venv/bin/python /tmp/issue13_cached_campaign.py \
  --classifiers emissary --class-counts 2 --seeds 42 \
  --train-per-class 70 --test-per-class 1 \
  --min-train-per-class 70 --min-test-per-class 1 --strict-support \
  --validation-fraction .2 --emissary-shots 0 5 100 \
  --emissary-shot-unit total \
  --emissary-mechanism project_fine_tuning \
  --emissary-project-id ms-dry-run-placeholder \
  --emissary-base-model Llama-3.2-1B-Instruct \
  --definitions class_definitions_data/banking77/canonical_llm_enriched_v1.json \
  --dry-run --output-root /tmp/emissary_project_sft_validation
```

## Live smoke

Authenticated read-only requests verified access to the Projects/model contract:
one project was visible, with no existing datasets, training jobs or deployments;
the selected Llama base model advertised classification support. No credentials
were logged.

After explicit authorization for two unpriced jobs and total-shot semantics, the
same 0/5/100 campaign was executed live:

- routing zero-shot completed two held-out predictions with accuracy and macro-F1
  1.0; this tiny sample is an execution check, not scientific evidence;
- the 5- and 100-shot JSONL datasets uploaded and profiled as classification;
- Emissary rejected each training submission with
  `Please setup your payment method first at the platform` in a JSON error body;
- a read-back confirmed zero training jobs and zero deployments, so no job was
  duplicated and no deployment remained active;
- provider usage/charge data was absent, so observed cost remains unavailable.

The live evidence directory is outside the repository at
`/tmp/emissary_project_sft_live/20260907T165914443762Z/`. The adapter now detects
and preserves successful-HTTP error bodies, and an offline regression test covers
the exact response observed.

Live fine-tuning acceptance remains open because the account has no payment method.
The two uploaded datasets and their content hashes are preserved in the evidence
for safe continuation. The smallest next action is configuring the payment method
in Emissary; the existing datasets can then be used without uploading them again.
