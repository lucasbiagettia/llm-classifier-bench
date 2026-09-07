# Emissary labeled-example contract investigation

Accessed **2026-09-06 America/Mexico_City** (2026-09-07 UTC). Baseline inspected:
`main` / `origin/main` at `b55a5fd6eae45ca1730be8ef089d80e1ac3c1eee`.
[Issue #13](https://github.com/lucasbiagettia/llm-classifier-bench/issues/13)
has no comments confirming total versus per-class 5/100 shots.
[PR #17](https://github.com/lucasbiagettia/llm-classifier-bench/pull/17)
already supplies the generic post-fit metadata hook; existing operational metrics
correctly represent missing costs as unavailable. Reused both.

## Confirmed experiment contract

The [experiment guide](https://docs.withemissary.com/experiment/) documents
`POST /v1/experiments` with `{name, mode, classes: [{name, description}]}`,
returning `id` and `latest_version`. Routing and decision use
`POST /v1/classification` with `{model: "EXPERIMENT_ID/VERSION", input,
data_format: "probs"}`. Authentication is `X-API-Key` against
`https://api.withemissary.com`. Pin the returned version once. Routing is the
existing benchmark mode; changing to decision would be a separate condition.

The guide describes uploading, labeling, and retraining as a subsequent workflow,
but gives no public operation or schema for that workflow on experiments.
The [API introduction](https://docs.withemissary.com/api/emissary-api/)
links the [published OpenAPI specification](https://github.com/Emissary-Tech/emissary-api-docs/blob/eed733e5637f43f75a97412c8ae107572db80de8/api/openapi.yaml),
version 0.1.0, downloaded SHA-256
`2f399ec3292385fff3320cd1ad2988908f9fc86123fe5bd4d31b1252107811e3`.
It exposes only POST creation under `/v1/experiments`, with no example field,
experiment training operation, job polling, or version lookup operation.
Creation response details come from the guide; the exported experiment operation
itself omits a response schema. Classification accepts no demonstrations.

## A different, documented training mechanism

The specification exposes a **project-based fine-tuning product**:

| Operation | Confirmed schema or behavior |
| --- | --- |
| POST `/v1/projects` | Creates project; separate identity from experiments |
| POST `/v1/projects/{project_id}/datasets` | Multipart `file`, optional `name`; upload limit 100 MB |
| POST `/v1/projects/{project_id}/training-jobs` | Requires `base_model`, `train_dataset_id`; `task_type="classification"`; optional hyperparameters and validation/split controls |
| GET `/v1/projects/{project_id}/training-jobs/{training_job_id}` | Pending, Running, Testing, Success, Failed, TimedOut, Cancelled statuses |
| GET corresponding `/checkpoints` | Enumerates training checkpoints |
| POST `/v1/projects/{project_id}/deployments` | Training job ID and checkpoint select the deployed model |
| GET corresponding deployment ID | Includes status, training ID, checkpoint; Deployed is a documented state |

The [dataset guide](https://docs.withemissary.com/fine-tuning/datasets/)
describes classification JSONL rows with `prompt` text and `completion` containing
every label mapped to 0 or 1. Multiclass has exactly one 1. The guide's approximately
20 examples per class is a **profiling warning**, not a confirmed API minimum.
No hard row-count maximum/minimum or missing-positive-class acceptance contract
was established. The upload byte limit belongs to project datasets, not an
experiment example endpoint.

The official [Python SDK](https://github.com/Emissary-Tech/emissary-python)
was inspected (README, SDK modules and models; source `_version.py` says 0.4.4).
It exposes project datasets, training jobs and deployments, with no experiment
retraining operation. Its project training support does not fill this gap.

**Scientific consequence:** project training requires selecting a base model;
the public sources do not establish that it is the routing experiment's model,
that class descriptions are preserved, or that its output has identical semantics.
Training another base model would confound shot count with model/product choice.
This implementation does not silently introduce that comparison.

## Costs, access and lifecycle gaps

The published schemas do not establish experiment training charges, usage units,
training duration, or a spending cap. Job creation/update timestamps do not measure
compute billing. Engine configuration describes idle deactivation and schedules;
reactivation may take 8–20 minutes. The specification includes cancellation/deletion
for project jobs, datasets, deployments, and engine controls, but no experiment
cleanup operation. These are not sufficient evidence for a bounded paid run.
The [pricing page](https://www.withemissary.com/pricing) returned a client-rendered
shell without verifiable rates in the retrieved content.

An API key is available locally (value never printed). That does not confirm
account entitlement to train the existing routing model. No private dashboard
endpoints, paid inference, training, or deployments were invoked.

## Decision and exact information needed

Nonzero experiment shots remain explicitly **unsupported**. Local planning can
select any feasible nonnegative budget; every live nonzero request fails before
remote creation. No fabricated upload payload, polling loop, job continuation,
training-time estimate, or asynchronous mock contract is shipped.

Emissary must supply a public routing-experiment labeled-example contract covering:

1. Endpoint/schema and permitted base model/experiment relationship, including
   class definitions and output probabilities; account entitlement requirements.
2. Total/per-class limits, missing-class behavior, and any automatic splitting,
   resampling, or data augmentation that changes the requested budget.
3. Submission IDs, readiness/failure states, timeout and continuation semantics,
   and exact immutable prediction version mapping.
4. Billable units/rates, usage/charge evidence and a bounded resource lifecycle.

Then implement and test those operations, confirm the intended shot unit with
Tanmay, and run the bounded held-out smoke comparison. Until then use `Refs #13`
and keep the PR draft. Offline mocks establish adapter behavior only; live
acceptance, training payloads, API sample-limit enforcement, asynchronous
readiness/failure/timeout and continuation remain open.
