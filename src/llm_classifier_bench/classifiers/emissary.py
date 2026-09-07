"""Thin, benchmark-oriented adapter for the public Emissary API."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic, perf_counter, sleep
from typing import Any, Callable, Mapping, Sequence, Self

import requests
from dotenv import load_dotenv

from llm_classifier_bench.config import EmissaryTrainingConfig
from llm_classifier_bench.datasets.selection import select_labeled_examples

from .base import (
    ClassificationInput,
    ClassDefinition,
    LabeledExample,
    Prediction,
)


DEFAULT_BASE_URL = "https://api.withemissary.com"


class EmissaryAPIError(RuntimeError):
    """Raised when Emissary returns a non-success HTTP response."""


class EmissaryResponseError(ValueError):
    """Raised when a successful response does not match the expected contract."""


class EmissaryPreparationError(RuntimeError):
    """Raised when an asynchronous preparation resource reaches failure."""


class EmissaryPreparationTimeout(TimeoutError):
    """Raised when bounded provider preparation does not finish in time."""


@dataclass(frozen=True, slots=True)
class Experiment:
    """Experiment created through ``POST /v1/experiments``."""

    experiment_id: str
    latest_version: str
    raw_response: Mapping[str, Any] | None = None

    @property
    def model_id(self) -> str:
        return f"{self.experiment_id}/{self.latest_version}"


class EmissaryClient:
    """Low-level HTTP client with no benchmark-specific policy hidden inside it.

    Inputs are sent exactly as received. The client intentionally does not
    truncate, compact, retry, or batch requests because those choices can alter
    benchmark semantics or latency measurements.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        load_dotenv(override=False)

        resolved_api_key = api_key or os.getenv("EMISSARY_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Missing Emissary API key. Set EMISSARY_API_KEY or pass api_key."
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = (connect_timeout_s, read_timeout_s)
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "X-API-Key": resolved_api_key,
            }
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                self._url(path),
                json=dict(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise EmissaryAPIError(f"Emissary request failed: {exc}") from exc

        return self._parse_json_response(response, path=path, expected_type=dict)

    def _get(self, path: str) -> dict[str, Any] | list[dict[str, Any]]:
        try:
            response = self.session.get(self._url(path), timeout=self.timeout)
        except requests.RequestException as exc:
            raise EmissaryAPIError(f"Emissary request failed: {exc}") from exc
        return self._parse_json_response(
            response,
            path=path,
            expected_type=(dict, list),
        )

    @staticmethod
    def _parse_json_response(
        response: requests.Response,
        *,
        path: str,
        expected_type: type | tuple[type, ...],
    ) -> Any:
        if not response.ok:
            body = response.text[:2_000]
            raise EmissaryAPIError(
                f"Emissary returned HTTP {response.status_code} for {path}: {body}"
            )

        try:
            parsed = response.json()
        except ValueError as exc:
            raise EmissaryResponseError(
                f"Emissary returned non-JSON content for {path}: "
                f"{response.text[:500]}"
            ) from exc

        if not isinstance(parsed, expected_type):
            if isinstance(expected_type, tuple):
                expected = " or ".join(item.__name__ for item in expected_type)
            else:
                expected = expected_type.__name__
            raise EmissaryResponseError(
                f"Expected JSON {expected} from {path}, got {type(parsed).__name__}"
            )
        return parsed

    def retrieve_base_model(self, model_name: str) -> dict[str, Any]:
        response = self._get(f"/v1/models/{model_name}")
        if not isinstance(response, dict):
            raise EmissaryResponseError("Base-model response must be an object")
        return response

    def retrieve_project(self, project_id: str) -> dict[str, Any]:
        response = self._get(f"/v1/projects/{project_id}")
        if not isinstance(response, dict):
            raise EmissaryResponseError("Project response must be an object")
        return response

    def upload_dataset(
        self,
        *,
        project_id: str,
        name: str,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        if not content:
            raise ValueError("Dataset content cannot be empty")
        if len(content) > 100_000_000:
            raise ValueError("Emissary dataset upload exceeds the documented 100 MB limit")
        path = f"/v1/projects/{project_id}/datasets"
        try:
            response = self.session.post(
                self._url(path),
                files={"file": (filename, content, "application/json")},
                data={"name": name},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise EmissaryAPIError(f"Emissary request failed: {exc}") from exc
        return self._parse_json_response(response, path=path, expected_type=dict)

    def retrieve_dataset(self, *, project_id: str, dataset_id: str) -> dict[str, Any]:
        response = self._get(f"/v1/projects/{project_id}/datasets/{dataset_id}")
        if not isinstance(response, dict):
            raise EmissaryResponseError("Dataset response must be an object")
        return response

    def create_training_job(
        self,
        *,
        project_id: str,
        base_model: str,
        train_dataset_id: str,
        name: str,
        parameters: Mapping[str, int | float | bool],
    ) -> dict[str, Any]:
        return self._post(
            f"/v1/projects/{project_id}/training-jobs",
            {
                "base_model_option": "pre-trained",
                "base_model": base_model,
                "train_dataset_id": train_dataset_id,
                "name": name,
                "task_type": "classification",
                "parameters": dict(parameters),
            },
        )

    def retrieve_training_job(
        self, *, project_id: str, training_job_id: str
    ) -> dict[str, Any]:
        response = self._get(
            f"/v1/projects/{project_id}/training-jobs/{training_job_id}"
        )
        if not isinstance(response, dict):
            raise EmissaryResponseError("Training-job response must be an object")
        return response

    def list_checkpoints(
        self, *, project_id: str, training_job_id: str
    ) -> list[dict[str, Any]]:
        response = self._get(
            f"/v1/projects/{project_id}/training-jobs/{training_job_id}/checkpoints"
        )
        if not isinstance(response, list) or any(not isinstance(item, dict) for item in response):
            raise EmissaryResponseError("Checkpoint response must be a list of objects")
        return response

    def create_deployment(
        self,
        *,
        project_id: str,
        training_job_id: str,
        checkpoint: int,
        name: str,
        inactive_timeout_s: int,
    ) -> dict[str, Any]:
        return self._post(
            f"/v1/projects/{project_id}/deployments",
            {
                "training_job_id": training_job_id,
                "checkpoint": checkpoint,
                "name": name,
                "description": "llm-classifier-bench labeled-example condition",
                "inactive_timeout": inactive_timeout_s,
            },
        )

    def retrieve_deployment(
        self, *, project_id: str, deployment_id: str
    ) -> dict[str, Any]:
        response = self._get(f"/v1/projects/{project_id}/deployments/{deployment_id}")
        if not isinstance(response, dict):
            raise EmissaryResponseError("Deployment response must be an object")
        return response

    def create_experiment(
        self,
        *,
        name: str,
        classes: Sequence[ClassDefinition],
        mode: str = "routing",
    ) -> Experiment:
        if len(classes) < 2:
            raise ValueError("An experiment requires at least two classes")

        class_names = [class_definition.name for class_definition in classes]
        if len(set(class_names)) != len(class_names):
            raise ValueError("Class names must be unique")

        response = self._post(
            "/v1/experiments",
            {
                "name": name,
                "mode": mode,
                "classes": [
                    {
                        "name": class_definition.name,
                        "description": class_definition.description,
                    }
                    for class_definition in classes
                ],
            },
        )

        experiment_id = response.get("id")
        latest_version = response.get("latest_version")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise EmissaryResponseError("Experiment response is missing a valid 'id'")
        if (not isinstance(latest_version, str) or not latest_version.strip()
                or latest_version.lower() == "latest" or "/" in latest_version):
            raise EmissaryResponseError(
                "Experiment response is missing a valid 'latest_version'"
            )

        return Experiment(
            experiment_id=experiment_id,
            latest_version=latest_version,
            raw_response=response,
        )

    def classify(
        self,
        *,
        model_id: str,
        text: str,
        data_format: str = "probs",
    ) -> dict[str, Any]:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if not text.strip():
            raise ValueError("text cannot be empty")

        return self._post(
            "/v1/classification",
            {
                "model": model_id,
                "input": text,
                "data_format": data_format,
            },
        )


def parse_classification_response(
    response: Mapping[str, Any],
    *,
    sample_id: str,
    latency_ms: float,
    probability_tolerance: float = 1e-4,
) -> Prediction:
    """Convert a raw Emissary response into the shared ``Prediction`` contract."""

    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise EmissaryResponseError("Classification response has no non-empty 'data' list")

    first_item = data[0]
    if not isinstance(first_item, Mapping):
        raise EmissaryResponseError("Classification response data[0] is not an object")

    raw_probabilities = first_item.get("probs")
    if not isinstance(raw_probabilities, Mapping) or not raw_probabilities:
        raise EmissaryResponseError(
            "Classification response data[0] has no non-empty 'probs' object"
        )

    probabilities: dict[str, float] = {}
    for label, value in raw_probabilities.items():
        if not isinstance(label, str) or not label:
            raise EmissaryResponseError("Probability labels must be non-empty strings")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EmissaryResponseError(
                f"Probability for label {label!r} is not numeric: {value!r}"
            )

        probability = float(value)
        if not math.isfinite(probability):
            raise EmissaryResponseError(
                f"Probability for label {label!r} is not finite"
            )
        if probability < 0.0 or probability > 1.0:
            raise EmissaryResponseError(
                f"Probability for label {label!r} is outside [0, 1]: {probability}"
            )
        probabilities[label] = probability

    probability_sum = sum(probabilities.values())
    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=probability_tolerance,
        abs_tol=probability_tolerance,
    ):
        raise EmissaryResponseError(
            f"Probabilities do not sum to 1 within tolerance: {probability_sum}"
        )

    predicted_label = max(probabilities, key=probabilities.__getitem__)
    confidence = probabilities[predicted_label]

    model = response.get("model")
    request_id = response.get("id")

    return Prediction(
        sample_id=sample_id,
        predicted_label=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
        latency_ms=latency_ms,
        model=model if isinstance(model, str) else None,
        request_id=request_id if isinstance(request_id, str) else None,
        raw_response=dict(response),
    )


def serialize_classification_dataset(
    examples: Sequence[LabeledExample],
    classes: Sequence[ClassDefinition],
) -> bytes:
    """Serialize the documented Emissary multiclass JSONL schema exactly."""

    frozen_classes = tuple(classes)
    class_names = [item.name for item in frozen_classes]
    if len(frozen_classes) < 2 or len(class_names) != len(set(class_names)):
        raise ValueError("Classification datasets require at least two unique classes")
    allowed = set(class_names)
    lines: list[str] = []
    for example in examples:
        if example.label not in allowed:
            raise ValueError(f"Unknown training label {example.label!r}")
        payload = {
            "prompt": example.text,
            "completion": {
                class_name: int(class_name == example.label)
                for class_name in class_names
            },
        }
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    if not lines:
        raise ValueError("A fine-tuning dataset cannot be empty")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _redact_download_urls(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: ("<redacted>" if key.endswith("_url") else _redact_download_urls(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_download_urls(item) for item in value]
    return value


class EmissaryClassifier:
    """Explicit zero-shot experiment or project SFT classification condition."""

    def __init__(
        self,
        *,
        client: EmissaryClient | None = None,
        training: EmissaryTrainingConfig | None = None,
        model_id: str | None = None,
        experiment_name: str | None = None,
        classifier_name: str = "emissary-zero-shot",
        sleep_fn: Callable[[float], None] = sleep,
        clock_fn: Callable[[], float] = monotonic,
    ) -> None:
        if model_id is not None and not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if model_id is not None:
            parts = model_id.split("/")
            if (
                len(parts) != 2
                or any(not part.strip() for part in parts)
                or parts[1].lower() == "latest"
            ):
                raise ValueError("model_id must pin an explicit experiment ID/version, not latest")
        if experiment_name is not None and not experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")
        if model_id is None and experiment_name is None:
            raise ValueError("Provide either model_id or experiment_name")
        if not classifier_name.strip():
            raise ValueError("classifier_name cannot be empty")

        self.training = training or EmissaryTrainingConfig()
        if self.training.shots and model_id is not None:
            raise ValueError("Nonzero shots cannot reuse an existing reference model")
        if not self.training.shots:
            self.supervision_regime = "zero_shot_routing_experiment"
        elif self.training.uses_project_fine_tuning:
            self.supervision_regime = "supervised_project_sft"
        else:
            self.supervision_regime = "labeled_examples_unresolved"
        self.training_examples_used = 0
        self.validation_examples_used = 0
        self._reference_model_id = model_id
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._metadata_sink: Callable[[Mapping[str, Any]], None] | None = None
        self._experiment: Experiment | None = None
        self._creation_ms: float | None = None
        self._preparation_ms: float | None = None
        self._preparation_started_at: float | None = None
        self._selection: dict[str, Any] | None = None
        self._dataset_payload_sha256: str | None = None
        self._dataset_payload_bytes: int | None = None
        self._dataset_id: str | None = None
        self._training_job_id: str | None = None
        self._checkpoint: int | None = None
        self._deployment_id: str | None = None
        self._deployment_name: str | None = None
        self._inference_model: str | None = model_id
        self._status = "ready" if model_id is not None else "not_prepared"
        self._last_error: dict[str, str] | None = None
        self._dataset_upload_ms: float | None = None
        self._training_submission_ms: float | None = None
        self._training_wall_ms: float | None = None
        self._deployment_submission_ms: float | None = None
        self._deployment_wall_ms: float | None = None
        self._base_model_response: dict[str, Any] | None = None
        self._project_response: dict[str, Any] | None = None
        self._dataset_response: dict[str, Any] | None = None
        self._training_response: dict[str, Any] | None = None
        self._checkpoint_responses: list[dict[str, Any]] = []
        self._deployment_response: dict[str, Any] | None = None
        self._training_status_history: list[dict[str, Any]] = []
        self._deployment_status_history: list[dict[str, Any]] = []
        self.client = client
        self.model_id = model_id
        self.experiment_name = experiment_name
        self._name = classifier_name
        self._classes: tuple[ClassDefinition, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        client: EmissaryClient,
        experiment_name: str,
        classes: Sequence[ClassDefinition],
        mode: str = "routing",
        classifier_name: str = "emissary-zero-shot",
    ) -> Self:
        classifier = cls(
            client=client, experiment_name=experiment_name, classifier_name=classifier_name,
        )
        started = perf_counter()
        experiment = client.create_experiment(
            name=experiment_name, classes=classes, mode=mode,
        )
        classifier.model_id = experiment.model_id
        classifier._reference_model_id = experiment.model_id
        classifier._classes = tuple(classes)
        classifier._experiment = experiment
        classifier._creation_ms = (perf_counter() - started) * 1000
        classifier._preparation_ms = classifier._creation_ms
        classifier._inference_model = experiment.model_id
        classifier._status = "ready"
        return classifier

    @property
    def name(self) -> str:
        return self._name

    def plan_fit(
        self, classes: Sequence[ClassDefinition], examples: Sequence[LabeledExample],
        *, validation_examples: Sequence[LabeledExample] = (),
    ) -> dict[str, Any]:
        """Resolve the exact fit selection without constructing an HTTP client."""
        selected, selection = select_labeled_examples(
            examples, classes, self.training, validation_examples=validation_examples,
        )
        unresolved = self.training.shots > 0 and not self.training.uses_project_fine_tuning
        if self.training.uses_project_fine_tuning:
            dataset = serialize_classification_dataset(selected, classes)
            selection["provider_dataset"] = {
                "format": "jsonl",
                "schema": "classification_label_map",
                "size_bytes": len(dataset),
                "sha256": sha256(dataset).hexdigest(),
            }
            operations = [
                "GET /v1/projects/{project_id}",
                f"GET /v1/models/{self.training.base_model}",
            ]
            if self.training.training_job_id is None:
                if self.training.dataset_id is None:
                    operations.extend(
                        [
                            "POST /v1/projects/{project_id}/datasets (multipart JSONL)",
                            "GET dataset until uploaded and profiled",
                        ]
                    )
                else:
                    operations.append("GET configured dataset until ready")
                operations.append(
                    "POST /v1/projects/{project_id}/training-jobs (SFT classification)"
                )
            operations.extend(
                [
                    "GET training job until Success or terminal failure",
                    "GET immutable training checkpoints",
                ]
            )
            if self.training.deployment_id is None:
                operations.append("POST /v1/projects/{project_id}/deployments")
            operations.extend(
                [
                    "GET deployment until Deployed or terminal failure",
                    "POST /v1/classification per held-out test example (probs)",
                ]
            )
        else:
            operations = [] if unresolved else [
                *([] if self._reference_model_id else ["POST /v1/experiments (routing)"]),
                "POST /v1/classification per held-out test example (probs)",
            ]
        return {
            "selection": selection,
            "live_supported": not unresolved,
            "mechanism": (
                "routing_experiment"
                if not self.training.shots
                else self.training.mechanism
            ),
            "scientific_comparability": (
                "Project SFT trains a separately selected base model and does not "
                "retrain the routing experiment; shot count is confounded with mechanism."
                if self.training.uses_project_fine_tuning
                else "Same routing-experiment zero-shot mechanism."
            ),
            "blocker": (
                "Select mechanism=project_fine_tuning explicitly for nonzero shots"
                if unresolved
                else None
            ),
            "planned_remote_operations": operations,
            "cost": {"available": False, "value_usd": None,
                     "reason": "Published API and model detail expose no price estimate"},
        }

    def fitted_metadata(self) -> dict[str, Any]:
        identity = self.model_id.rsplit("/", 1) if self.model_id else []
        return {
            "status": self._status,
            "mechanism": (
                "project_fine_tuning"
                if self.training.uses_project_fine_tuning
                else "routing_experiment"
            ),
            "selection": self._selection,
            "training_examples_used": self.training_examples_used,
            "validation_examples_used": self.validation_examples_used,
            "model_id": self.model_id,
            "experiment_id": identity[0] if len(identity) == 2 else None,
            "model_version": identity[1] if len(identity) == 2 else None,
            "project_id": self.training.project_id,
            "base_model": self.training.base_model,
            "dataset_id": self._dataset_id,
            "training_job_id": self._training_job_id,
            "checkpoint": self._checkpoint,
            "deployment_id": self._deployment_id,
            "deployment_name": self._deployment_name,
            "inference_model": self._inference_model,
            "dataset_payload_sha256": self._dataset_payload_sha256,
            "dataset_payload_bytes": self._dataset_payload_bytes,
            "experiment_creation_ms": self._creation_ms,
            "example_submission_ms": self._dataset_upload_ms,
            "training_submission_ms": self._training_submission_ms,
            "training_wall_ms": self._training_wall_ms,
            "deployment_submission_ms": self._deployment_submission_ms,
            "deployment_wall_ms": self._deployment_wall_ms,
            "preparation_wall_ms": self._preparation_ms,
            "provider_training_ms": None,
            "training_timing_reason": (
                "Provider does not report training-only duration; wall time includes queue/testing"
                if self.training.uses_project_fine_tuning
                else "Zero-shot condition does not submit training"
            ),
            "last_error": self._last_error,
            "creation_response": (
                dict(self._experiment.raw_response or {}) if self._experiment else None
            ),
            "project_response": _redact_download_urls(self._project_response),
            "base_model_response": _redact_download_urls(self._base_model_response),
            "dataset_response": _redact_download_urls(self._dataset_response),
            "training_response": _redact_download_urls(self._training_response),
            "checkpoint_responses": _redact_download_urls(self._checkpoint_responses),
            "deployment_response": _redact_download_urls(self._deployment_response),
            "training_status_history": _redact_download_urls(self._training_status_history),
            "deployment_status_history": _redact_download_urls(self._deployment_status_history),
            "cost": {"available": False, "value_usd": None,
                     "reason": "No price or charge returned by the documented preparation API"},
        }

    def set_fit_metadata_sink(
        self, sink: Callable[[Mapping[str, Any]], None] | None
    ) -> None:
        """Let the runner persist continuation IDs after each remote mutation."""

        self._metadata_sink = sink

    def _persist_progress(self) -> None:
        if self._metadata_sink is not None:
            self._metadata_sink(self.fitted_metadata())

    def _reset_runtime_state(self) -> None:
        self.training_examples_used = 0
        self.validation_examples_used = 0
        self._selection = None
        self._dataset_payload_sha256 = None
        self._dataset_payload_bytes = None
        self._dataset_id = self.training.dataset_id
        self._training_job_id = self.training.training_job_id
        self._checkpoint = self.training.checkpoint
        self._deployment_id = self.training.deployment_id
        self._deployment_name = None
        self._inference_model = self._reference_model_id
        self._status = "not_prepared"
        self._last_error = None
        self._dataset_upload_ms = None
        self._training_submission_ms = None
        self._training_wall_ms = None
        self._deployment_submission_ms = None
        self._deployment_wall_ms = None
        self._base_model_response = None
        self._project_response = None
        self._dataset_response = None
        self._training_response = None
        self._checkpoint_responses = []
        self._deployment_response = None
        self._training_status_history = []
        self._deployment_status_history = []

    def prepare(self, classes: Sequence[ClassDefinition]) -> None:
        """Configure classes and create an experiment when one was not pre-created."""

        self._preparation_started_at = perf_counter()
        self._reset_runtime_state()
        self.model_id = None
        self._classes = ()
        if self._reference_model_id is None:
            self._experiment = None
            self._creation_ms = None
            self._preparation_ms = None
        self.training.require_live_support()
        frozen = tuple(classes)
        if len(frozen) < 2:
            raise ValueError("At least two classes are required")
        if len({item.name for item in frozen}) != len(frozen):
            raise ValueError("Class names must be unique")
        self._classes = frozen
        if self.training.uses_project_fine_tuning:
            self._status = "classes_prepared"
            return
        self.model_id = self._reference_model_id

        if self.model_id is None:
            if self.experiment_name is None:  # defensive; constructor prevents this
                raise RuntimeError("No experiment_name is available")
            if self.client is None:
                self.client = EmissaryClient()
            started = perf_counter()
            experiment = self.client.create_experiment(
                name=self.experiment_name,
                classes=frozen,
                mode="routing",
            )
            self.model_id = experiment.model_id
            self._experiment = experiment
            self._creation_ms = (perf_counter() - started) * 1000
        self._preparation_ms = perf_counter() - self._preparation_started_at
        self._preparation_ms *= 1_000
        self._inference_model = self.model_id
        self._status = "ready"

    def fit(
        self,
        examples: Sequence[LabeledExample],
        *,
        validation_examples: Sequence[LabeledExample] = (),
    ) -> None:
        """Select examples and run the configured provider preparation flow."""
        started = perf_counter()
        self._selection = None
        self.training.require_live_support()
        if not self._classes:
            raise RuntimeError("Call prepare(classes) before fit()")
        selected, self._selection = select_labeled_examples(
            examples,
            self._classes,
            self.training,
            validation_examples=validation_examples,
        )
        self.training_examples_used = len(selected)
        if not self.training.uses_project_fine_tuning:
            if self._preparation_ms is not None:
                self._preparation_ms += (perf_counter() - started) * 1_000
            self._status = "ready"
            return
        try:
            self._fit_project(selected)
            self._status = "ready"
        except Exception as exc:
            self._status = "failed"
            self._last_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self._persist_progress()
            raise
        finally:
            if self._preparation_started_at is not None:
                self._preparation_ms = (
                    perf_counter() - self._preparation_started_at
                ) * 1_000

    def _fit_project(self, selected: Sequence[LabeledExample]) -> None:
        if self.client is None:
            self.client = EmissaryClient()
        project_id = self.training.project_id
        base_model = self.training.base_model
        if project_id is None or base_model is None:  # validated by config
            raise RuntimeError("Project fine-tuning configuration is incomplete")

        self._status = "validating_project"
        project_response = self.client.retrieve_project(project_id)
        self._project_response = project_response
        if project_response.get("id") != project_id:
            raise EmissaryResponseError("Project response does not match configured project")

        self._status = "validating_base_model"
        base_response = self.client.retrieve_base_model(base_model)
        self._base_model_response = base_response
        if base_response.get("name") != base_model:
            raise EmissaryResponseError("Base-model response does not match configured model")
        supported = base_response.get("support_task")
        if not isinstance(supported, list) or "classification" not in supported:
            raise ValueError(f"Base model {base_model!r} does not support classification")

        dataset_payload = serialize_classification_dataset(selected, self._classes)
        self._dataset_payload_sha256 = sha256(dataset_payload).hexdigest()
        self._dataset_payload_bytes = len(dataset_payload)
        if (
            self.training.dataset_sha256 is not None
            and self.training.dataset_sha256 != self._dataset_payload_sha256
        ):
            raise ValueError(
                "Selected examples do not match the configured continuation dataset_sha256"
            )

        if self._dataset_id is None:
            self._status = "uploading_dataset"
            upload_started = perf_counter()
            response = self.client.upload_dataset(
                project_id=project_id,
                name=f"{self.experiment_name}-dataset",
                filename=f"{self.experiment_name}.jsonl",
                content=dataset_payload,
            )
            self._dataset_upload_ms = (perf_counter() - upload_started) * 1_000
            self._dataset_id = self._require_id(response, resource="dataset")
            self._dataset_response = response
            self._persist_progress()
        self._dataset_response = self._wait_for_dataset(project_id, self._dataset_id)

        if self._training_job_id is None:
            self._status = "submitting_training"
            submission_started = perf_counter()
            response = self.client.create_training_job(
                project_id=project_id,
                base_model=base_model,
                train_dataset_id=self._dataset_id,
                name=f"{self.experiment_name}-training",
                parameters=self.training.training_parameters(),
            )
            self._training_submission_ms = (
                perf_counter() - submission_started
            ) * 1_000
            self._training_job_id = self._require_id(response, resource="training job")
            self._training_response = response
            self._persist_progress()

        training_started = perf_counter()
        try:
            self._training_response = self._wait_for_training(
                project_id,
                self._training_job_id,
                initial=None,
            )
        finally:
            self._training_wall_ms = (perf_counter() - training_started) * 1_000
        self._validate_training_identity(self._training_response)

        checkpoints = self.client.list_checkpoints(
            project_id=project_id,
            training_job_id=self._training_job_id,
        )
        self._checkpoint_responses = checkpoints
        available = {
            item.get("checkpoint")
            for item in checkpoints
            if type(item.get("checkpoint")) is int and item["checkpoint"] >= 0
        }
        if not available:
            raise EmissaryResponseError("Successful training job returned no checkpoints")
        if self._checkpoint is None:
            self._checkpoint = max(available)
        elif self._checkpoint not in available:
            raise ValueError(
                f"Configured checkpoint {self._checkpoint} is unavailable; "
                f"available={sorted(available)}"
            )

        deployment_started = perf_counter()
        if self._deployment_id is None:
            self._status = "submitting_deployment"
            submission_started = perf_counter()
            response = self.client.create_deployment(
                project_id=project_id,
                training_job_id=self._training_job_id,
                checkpoint=self._checkpoint,
                name=f"{self.experiment_name}-deployment",
                inactive_timeout_s=self.training.inactive_timeout_s,
            )
            self._deployment_submission_ms = (
                perf_counter() - submission_started
            ) * 1_000
            self._deployment_id = self._require_id(response, resource="deployment")
            name = response.get("name")
            self._deployment_name = name if isinstance(name, str) and name else None
            self._deployment_response = response
            self._persist_progress()

        try:
            self._deployment_response = self._wait_for_deployment(
                project_id,
                self._deployment_id,
                initial=None,
            )
        finally:
            self._deployment_wall_ms = (perf_counter() - deployment_started) * 1_000
        self._validate_deployment_identity(self._deployment_response)
        self.model_id = self._deployment_id
        self._inference_model = self._deployment_name
        self._persist_progress()

    @staticmethod
    def _require_id(response: Mapping[str, Any], *, resource: str) -> str:
        resource_id = response.get("id")
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise EmissaryResponseError(f"{resource} response is missing a valid id")
        return resource_id

    def _wait_for_dataset(self, project_id: str, dataset_id: str) -> dict[str, Any]:
        deadline = self._clock() + self.training.dataset_timeout_s
        while True:
            self._status = "waiting_for_dataset"
            response = self.client.retrieve_dataset(
                project_id=project_id,
                dataset_id=dataset_id,
            )
            self._dataset_response = response
            self._persist_progress()
            if response.get("id") != dataset_id:
                raise EmissaryResponseError("Dataset response ID changed while polling")
            if response.get("is_uploaded") is True and response.get("is_profiled") is True:
                compatible = response.get("compatible_task_types")
                if not isinstance(compatible, list) or "classification" not in compatible:
                    raise EmissaryPreparationError(
                        "Profiled dataset is not compatible with classification"
                    )
                return response
            self._sleep_until_next_poll(deadline, resource="dataset")

    def _wait_for_training(
        self,
        project_id: str,
        training_job_id: str,
        *,
        initial: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        deadline = self._clock() + self.training.training_timeout_s
        response = dict(initial) if initial is not None else None
        while True:
            self._status = "waiting_for_training"
            if response is None:
                response = self.client.retrieve_training_job(
                    project_id=project_id,
                    training_job_id=training_job_id,
                )
            if response.get("id") != training_job_id:
                raise EmissaryResponseError("Training-job response ID changed while polling")
            status = response.get("status")
            self._training_status_history.append(dict(response))
            self._persist_progress()
            if status == "Success":
                return response
            if status in {"Failed", "TimedOut", "Cancelled"}:
                raise EmissaryPreparationError(
                    f"Training job {training_job_id} reached terminal status {status}"
                )
            if status not in {"Pending", "Running", "Testing"}:
                raise EmissaryResponseError(f"Unknown training-job status {status!r}")
            self._sleep_until_next_poll(deadline, resource="training job")
            response = None

    def _wait_for_deployment(
        self,
        project_id: str,
        deployment_id: str,
        *,
        initial: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        deadline = self._clock() + self.training.deployment_timeout_s
        response = dict(initial) if initial is not None else None
        while True:
            self._status = "waiting_for_deployment"
            if response is None:
                response = self.client.retrieve_deployment(
                    project_id=project_id,
                    deployment_id=deployment_id,
                )
            if response.get("id") != deployment_id:
                raise EmissaryResponseError("Deployment response ID changed while polling")
            status = response.get("status")
            self._deployment_status_history.append(dict(response))
            self._persist_progress()
            if status == "Deployed":
                return response
            if status in {
                "Failed",
                "Cancelled",
                "Terminated",
                "Deactivated",
                "TimedOut",
            }:
                raise EmissaryPreparationError(
                    f"Deployment {deployment_id} reached terminal status {status}"
                )
            if status not in {"Pending", "Deploying", "Reactivating"}:
                raise EmissaryResponseError(f"Unknown deployment status {status!r}")
            self._sleep_until_next_poll(deadline, resource="deployment")
            response = None

    def _sleep_until_next_poll(self, deadline: float, *, resource: str) -> None:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise EmissaryPreparationTimeout(
                f"Timed out waiting for Emissary {resource} readiness"
            )
        self._sleep(min(self.training.poll_interval_s, remaining))

    def _validate_training_identity(self, response: Mapping[str, Any]) -> None:
        if response.get("task_type") != "classification":
            raise EmissaryResponseError("Training job is not a classification job")
        if response.get("base_model") != self.training.base_model:
            raise EmissaryResponseError("Training job base model differs from configuration")
        train_dataset = response.get("train_dataset")
        if self._dataset_id is not None:
            if (
                not isinstance(train_dataset, Mapping)
                or train_dataset.get("id") != self._dataset_id
            ):
                raise EmissaryResponseError("Training job dataset differs from configuration")

    def _validate_deployment_identity(self, response: Mapping[str, Any]) -> None:
        if response.get("training_id") != self._training_job_id:
            raise EmissaryResponseError("Deployment training ID differs from pinned job")
        if response.get("checkpoint") != self._checkpoint:
            raise EmissaryResponseError("Deployment checkpoint differs from pinned checkpoint")
        if response.get("task_type") != "classification":
            raise EmissaryResponseError("Deployment is not a classification deployment")
        if response.get("base_model") != self.training.base_model:
            raise EmissaryResponseError("Deployment base model differs from configuration")
        name = response.get("name")
        if not isinstance(name, str) or not name.strip():
            raise EmissaryResponseError("Deployment response is missing its inference name")
        self._deployment_name = name
        metadata = response.get("metadata")
        labels = metadata.get("labels") if isinstance(metadata, Mapping) else None
        if not isinstance(labels, list) or set(labels) != {item.name for item in self._classes}:
            raise EmissaryResponseError(
                "Deployment labels must contain exactly the configured class set"
            )

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        self.training.require_live_support()
        if self._status != "ready" or self.model_id is None or self._inference_model is None:
            raise RuntimeError("Call prepare(classes) and fit(examples) before predict()")

        if self.client is None:
            self.client = EmissaryClient()
        predictions: list[Prediction] = []

        for example in examples:
            started_at = perf_counter()
            response = self.client.classify(
                model_id=self._inference_model,
                text=example.text,
                data_format="probs",
            )
            latency_ms = (perf_counter() - started_at) * 1_000

            prediction = parse_classification_response(
                response, sample_id=example.sample_id, latency_ms=latency_ms,
            )
            if self._classes and set(prediction.probabilities or {}) != {
                item.name for item in self._classes
            }:
                raise EmissaryResponseError(
                    "Probabilities must contain exactly the configured class set"
                )
            if self.training.uses_project_fine_tuning:
                if prediction.model not in {self._deployment_id, self._deployment_name}:
                    raise EmissaryResponseError(
                        "Response model differs from the pinned deployment"
                    )
            elif prediction.model is not None and prediction.model != self.model_id:
                raise EmissaryResponseError(
                    "Response model differs from the pinned model version"
                )
            predictions.append(prediction)

        return predictions
