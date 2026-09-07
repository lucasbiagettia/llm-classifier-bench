# Emissary labeled-example contract

Investigated 2026-09-06/07 against the public documentation, published OpenAPI,
official Python SDK and authenticated read-only API responses. No training job,
dataset, deployment or paid inference was created during the investigation.

## Two separate mechanisms

The routing experiment API remains zero-shot. `POST /v1/experiments` accepts a
name, mode and class names/descriptions; `/v1/classification` accepts a model and
input. Neither the documented experiment request nor classification request has
an examples field, and the public API exposes no experiment retraining endpoint.

Emissary also exposes a documented Projects fine-tuning product. Issue #13 now
supports this product only when the caller selects it explicitly:

| Stage | Public operation and verified requirement |
| --- | --- |
| Project | `GET /v1/projects/{project_id}` validates an existing project |
| Base model | `GET /v1/models/{model}` must advertise classification support |
| Dataset | multipart `POST /v1/projects/{project_id}/datasets`; maximum 100 MB |
| Profiling | dataset GET until `is_uploaded` and `is_profiled`; classification must be compatible |
| Training | `POST .../training-jobs` with base model, dataset ID, `task_type=classification` and parameters |
| Readiness | training GET: Pending, Running, Testing, Success, Failed, TimedOut or Cancelled |
| Checkpoint | job `/checkpoints` GET returns integer checkpoints |
| Deployment | deployment POST pins the training job and checkpoint |
| Serving | deployment GET must reach Deployed, then `/v1/classification` uses its name |

The documented classification dataset is JSONL. Each row contains `prompt` and a
`completion` object mapping every label to 0 or 1, with exactly one positive label
for multiclass data.

The public sources describe roughly 20 examples per class as a profiling warning,
not a hard request constraint. The implementation therefore preserves the
requested budget and lets profiling report compatibility instead of silently
inflating 5-shot data.

## Live observations

Authenticated read-only calls confirmed one accessible project and no existing
datasets, jobs or deployments. `/v1/models` returned 39 models. The inspected
`Llama-3.2-1B-Instruct` and `Llama-3.2-3B-Instruct` entries advertise
classification support; the former exposes defaults including three epochs,
learning rate 0.0002, train batch size 2 and evaluation batch size 1.

Those observations validate entitlement to read these resources. During the
authorized live smoke, both selected JSONL datasets uploaded and profiled as
classification, but training creation returned a successful HTTP response with
`Please setup your payment method first at the platform`. A subsequent list call
confirmed that no training job or deployment existed. The adapter preserves and
reports these provider error bodies. Training remains blocked until the account
has a payment method.

## Scientific and operational limits

Projects fine-tuning trains a separately chosen base model. Public sources do not
state that this model is the routing experiment's model or that class descriptions
are preserved. A comparison between Emissary routing zero-shot and Projects SFT
therefore confounds shot budget, product mechanism and base model. Artifacts label
the regimes `zero_shot_routing_experiment` and `supervised_project_sft` and record
this limitation.

The public API and inspected model detail expose no training price or enforceable
USD cap. Cost remains unavailable rather than zero. Live campaign execution needs
an explicit unpriced-training acknowledgement and a maximum count of newly
created jobs, and the account must have a payment method configured.

Submission timeouts are ambiguous and are never retried automatically. Polling
has bounded deadlines and explicit terminal failures. Remote IDs and sanitized
responses are persisted after each mutation. Resuming verifies the locally
recreated JSONL hash, dataset ID, job/base/task identity, checkpoint, deployment
identity and exact output labels before held-out inference.

References: [experiment guide](https://docs.withemissary.com/experiment/),
[API introduction](https://docs.withemissary.com/api/emissary-api/),
[classification dataset guide](https://docs.withemissary.com/fine-tuning/datasets/),
[published OpenAPI](https://github.com/Emissary-Tech/emissary-api-docs/blob/eed733e5637f43f75a97412c8ae107572db80de8/api/openapi.yaml),
and [official Python SDK](https://github.com/Emissary-Tech/emissary-python).
