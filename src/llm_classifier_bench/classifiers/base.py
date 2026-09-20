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
    full distribution. Emissary and the supervised local classifiers do; the OpenAI
    generative baseline intentionally does not fabricate one.

    ``latency_ms`` is the adapter's internal observation. The runner preserves it
    as ``adapter_latency_ms`` in JSONL and records its common outer-call boundary
    separately; direct workflow consumers still see the adapter observation.
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
    """Behavior required by the benchmark runner.

    Lifecycle:

    ``prepare(classes) -> fit(train, validation) -> predict(test)``

    Implementations may optionally expose ``fitted_metadata()`` returning a
    JSON-compatible mapping; the runner saves it as ``fit_metadata.json`` after fit.

    An optional ``plan_fit(classes, train, validation_examples=...)`` hook must
    perform no remote operations; the runner saves its result before prepare.
    Dry-run execution requires that hook and stops before the lifecycle begins.
    A remote classifier may expose ``set_fit_metadata_sink(callable)`` so the
    runner can durably save continuation identifiers between fit milestones.

    ``prepare`` communicates the closed label space without leaking labeled
    examples. ``fit`` may be a no-op for zero-shot classifiers.

    Optional ``inference_metadata()`` returns JSON-compatible backend, device,
    batching, timeout and retry settings without making provider requests. Unknown
    transport-attempt timing or hardware must remain explicitly unavailable.

    Optional ``set_usage_sink(callable)`` records each API attempt, including
    responses rejected during parsing, for runner-owned inference accounting.
    This hook is separate from Prediction because failed calls have costs too.

    Optional ``set_preparation_sink(callable)`` installs a stage-context factory
    during prepare/fit. ``preparation_metadata()`` may expose resource/cache
    provenance and documented billing evidence; missing provider prices stay unknown.
    """

    @property
    def name(self) -> str:
        """Stable human-readable classifier name."""
        ...

    def prepare(self, classes: Sequence[ClassDefinition]) -> None:
        """Configure the closed label space used by subsequent predictions."""
        ...

    def fit(
        self,
        examples: Sequence[LabeledExample],
        *,
        validation_examples: Sequence[LabeledExample] = (),
    ) -> None:
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
