# Emissary labeled-example contract

This describes the request/response contract implemented by the adapter.
Account readiness, model availability and billing must be checked for each live
campaign; a successful dry run does not establish provider acceptance.

## Two separate mechanisms

The routing experiment API remains zero-shot. `POST /v1/experiments` accepts a
name, mode and class names/descriptions; `/v1/classification` accepts a model and
input. Neither the documented experiment request nor classification request has
an examples field, and the public API exposes no experiment retraining endpoint.

The adapter also supports the Projects fine-tuning product when the caller
selects it explicitly:

| Stage | Operation expected by the adapter |
| --- | --- |
| Project | `GET /v1/projects/{project_id}` validates an existing project |
| Base model | `GET /v1/models/{model}` must advertise classification support |
| Dataset | multipart `POST /v1/projects/{project_id}/datasets` |
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

## Scientific and operational limits

Projects fine-tuning trains a separately chosen base model. Public sources do not
state that this model is the routing experiment's model or that class descriptions
are preserved. A comparison between Emissary routing zero-shot and Projects SFT
therefore confounds shot budget, product mechanism and base model. Artifacts label
the regimes `zero_shot_routing_experiment` and `supervised_project_sft` and record
this limitation.

The adapter has no training-price estimate or enforceable USD cap.
Cost remains unavailable rather than zero. Live campaign execution needs
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
