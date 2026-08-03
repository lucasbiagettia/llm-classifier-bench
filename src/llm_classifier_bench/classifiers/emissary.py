"""Thin, benchmark-oriented adapter for the public Emissary API."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence, Self

import requests
from dotenv import load_dotenv

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
        if not isinstance(latest_version, str) or not latest_version:
            raise EmissaryResponseError(
                "Experiment response is missing a valid 'latest_version'"
            )

        return Experiment(
            experiment_id=experiment_id,
            latest_version=latest_version,
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
        client: EmissaryClient,
        model_id: str | None = None,
        experiment_name: str | None = None,
        classifier_name: str = "emissary-zero-shot",
    ) -> None:
        if model_id is not None and not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if experiment_name is not None and not experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")
        if model_id is None and experiment_name is None:
            raise ValueError("Provide either model_id or experiment_name")
        if not classifier_name.strip():
            raise ValueError("classifier_name cannot be empty")

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
        experiment = client.create_experiment(
            name=experiment_name,
            classes=classes,
            mode=mode,
        )
        return cls(
            client=client,
            model_id=experiment.model_id,
            experiment_name=experiment_name,
            classifier_name=classifier_name,
        )

    @property
    def name(self) -> str:
        return self._name

    def prepare(self, classes: Sequence[ClassDefinition]) -> None:
        """Configure classes and create an experiment when one was not pre-created."""

        frozen = tuple(classes)
        if len(frozen) < 2:
            raise ValueError("At least two classes are required")
        self._classes = frozen

        if self.model_id is None:
            if self.experiment_name is None:  # defensive; constructor prevents this
                raise RuntimeError("No experiment_name is available")
            experiment = self.client.create_experiment(
                name=self.experiment_name,
                classes=frozen,
                mode="routing",
            )
            self.model_id = experiment.model_id

    def fit(
        self,
        examples: Sequence[LabeledExample],
        *,
        validation_examples: Sequence[LabeledExample] = (),
    ) -> None:
        """Zero-shot Emissary requires no local fitting."""
        return None

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        if self.model_id is None:
            raise RuntimeError("Call prepare(classes) before predict()")

        predictions: list[Prediction] = []

        for example in examples:
            started_at = perf_counter()
            response = self.client.classify(
                model_id=self.model_id,
                text=example.text,
                data_format="probs",
            )
            latency_ms = (perf_counter() - started_at) * 1_000

            predictions.append(
                parse_classification_response(
                    response,
                    sample_id=example.sample_id,
                    latency_ms=latency_ms,
                )
            )

        return predictions
