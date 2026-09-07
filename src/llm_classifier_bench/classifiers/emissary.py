"""Thin, benchmark-oriented adapter for the public Emissary API."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence, Self

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
                "Content-Type": "application/json",
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
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise EmissaryAPIError(f"Emissary request failed: {exc}") from exc

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

        if not isinstance(parsed, dict):
            raise EmissaryResponseError(
                f"Expected a JSON object from {path}, got {type(parsed).__name__}"
            )
        return parsed

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


class EmissaryClassifier:
    """Zero-shot Emissary classifier backed by an experiment model version."""

    def __init__(
        self,
        *,
        client: EmissaryClient | None = None,
        training: EmissaryTrainingConfig | None = None,
        model_id: str | None = None,
        experiment_name: str | None = None,
        classifier_name: str = "emissary-zero-shot",
    ) -> None:
        if model_id is not None and not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if model_id is not None:
            parts = model_id.split("/")
            if len(parts) != 2 or any(not part.strip() for part in parts) or parts[1].lower() == "latest":
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
        self.supervision_regime = "zero_shot" if not self.training.shots else "labeled_examples_unsupported"
        self.training_examples_used = 0
        self.validation_examples_used = 0
        self._reference_model_id = model_id
        self._experiment: Experiment | None = None
        self._creation_ms: float | None = None
        self._preparation_ms: float | None = None
        self._selection: dict[str, Any] | None = None
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
        return classifier

    @property
    def name(self) -> str:
        return self._name

    def plan_fit(
        self, classes: Sequence[ClassDefinition], examples: Sequence[LabeledExample],
        *, validation_examples: Sequence[LabeledExample] = (),
    ) -> dict[str, Any]:
        """Resolve the exact fit selection without constructing an HTTP client."""
        _, selection = select_labeled_examples(
            examples, classes, self.training, validation_examples=validation_examples,
        )
        return {
            "selection": selection,
            "live_supported": self.training.shots == 0,
            "blocker": None if not self.training.shots else (
                "Public routing-experiment labeled-example/retraining contract unavailable"
            ),
            "planned_remote_operations": [] if self.training.shots else [
                *([] if self._reference_model_id else ["POST /v1/experiments (routing)"]),
                "POST /v1/classification per held-out test example (probs)",
            ],
            "cost": {"available": False, "value_usd": None,
                     "reason": "No verified account-specific pricing or charge estimate"},
        }

    def fitted_metadata(self) -> dict[str, Any]:
        identity = self.model_id.rsplit("/", 1) if self.model_id else []
        return {
            "status": "ready" if self.model_id else "not_prepared",
            "selection": self._selection,
            "model_id": self.model_id,
            "experiment_id": identity[0] if len(identity) == 2 else None,
            "model_version": identity[1] if len(identity) == 2 else None,
            "job_id": None,
            "experiment_creation_ms": self._creation_ms,
            "example_submission_ms": None,
            "preparation_wall_ms": self._preparation_ms,
            "provider_training_ms": None,
            "training_timing_reason": "Zero-shot condition does not submit training",
            "creation_response": dict(self._experiment.raw_response or {}) if self._experiment else None,
            "cost": {"available": False, "value_usd": None,
                     "reason": "No verified pricing; raw provider evidence retained when returned"},
        }

    def prepare(self, classes: Sequence[ClassDefinition]) -> None:
        """Configure classes and create an experiment when one was not pre-created."""

        preparation_started = perf_counter()
        self._selection = None
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
        self._preparation_ms = (perf_counter() - preparation_started) * 1000

    def fit(
        self,
        examples: Sequence[LabeledExample],
        *,
        validation_examples: Sequence[LabeledExample] = (),
    ) -> None:
        """Record selection evidence; never simulate unsupported training."""
        started = perf_counter()
        self._selection = None
        self.training.require_live_support()
        if self._classes:
            _, self._selection = select_labeled_examples(
                examples, self._classes, self.training,
                validation_examples=validation_examples,
            )
        if self._preparation_ms is not None:
            self._preparation_ms += (perf_counter() - started) * 1000

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        self.training.require_live_support()
        if self.model_id is None:
            raise RuntimeError("Call prepare(classes) before predict()")

        if self.client is None:
            self.client = EmissaryClient()
        predictions: list[Prediction] = []

        for example in examples:
            started_at = perf_counter()
            response = self.client.classify(
                model_id=self.model_id,
                text=example.text,
                data_format="probs",
            )
            latency_ms = (perf_counter() - started_at) * 1_000

            prediction = parse_classification_response(
                response, sample_id=example.sample_id, latency_ms=latency_ms,
            )
            if self._classes and set(prediction.probabilities or {}) != {item.name for item in self._classes}:
                raise EmissaryResponseError("Probabilities must contain exactly the configured class set")
            if prediction.model is not None and prediction.model != self.model_id:
                raise EmissaryResponseError("Response model differs from the pinned model version")
            predictions.append(prediction)

        return predictions
