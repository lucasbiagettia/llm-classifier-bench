"""Shared classifier contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from llm_classifier_bench.core import (
    ClassificationInput,
    ClassDefinition,
    LabeledExample,
)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Normalized prediction returned by every classifier implementation.

    ``probabilities`` is optional because not every classification API exposes a
    full distribution.
    """

    sample_id: str
    predicted_label: str
    confidence: float | None
    probabilities: Mapping[str, float] | None
    latency_ms: float
    model: str | None = None
    request_id: str | None = None
    raw_response: Mapping[str, Any] | None = None


@runtime_checkable
class Classifier(Protocol):
    """Minimal behavior required by the future benchmark runner."""

    @property
    def name(self) -> str:
        """Stable human-readable classifier name."""
        ...

    def fit(self, examples: Sequence[LabeledExample]) -> None:
        """Fit the classifier, or do nothing for zero-shot/API classifiers."""
        ...

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        """Predict one result per input, preserving input order."""
        ...


__all__ = [
    "ClassificationInput",
    "ClassDefinition",
    "Classifier",
    "LabeledExample",
    "Prediction",
]
