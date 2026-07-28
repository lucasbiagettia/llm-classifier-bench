"""Shared contracts and normalized records for benchmark metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Classifier output normalized for metric computation.

    This object deliberately sits after the dataset/classifier interfaces. It can
    be built either from an in-memory ``ClassificationRecord`` or from a saved
    JSONL artifact, so metrics never need to call a model again.
    """

    sample_id: str
    gold_label: str
    predicted_label: str
    confidence: float | None
    probabilities: Mapping[str, float] | None
    latency_ms: float | None
    cost_usd: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("EvaluationRecord.sample_id cannot be empty")
        if not self.gold_label.strip():
            raise ValueError("EvaluationRecord.gold_label cannot be empty")
        if not self.predicted_label.strip():
            raise ValueError("EvaluationRecord.predicted_label cannot be empty")

        if self.confidence is not None:
            if not math.isfinite(self.confidence):
                raise ValueError("EvaluationRecord.confidence must be finite")
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("EvaluationRecord.confidence must be between 0 and 1")

        if self.latency_ms is not None:
            if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
                raise ValueError("EvaluationRecord.latency_ms must be finite and non-negative")

        if self.cost_usd is not None:
            if not math.isfinite(self.cost_usd) or self.cost_usd < 0.0:
                raise ValueError("EvaluationRecord.cost_usd must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One scalar benchmark result plus auditable calculation metadata."""

    name: str
    value: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "available": self.available,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def unavailable(
        cls,
        *,
        name: str,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MetricResult":
        details = dict(metadata or {})
        details["reason"] = reason
        return cls(name=name, value=None, metadata=details)


@runtime_checkable
class Metric(Protocol):
    """Minimal interface implemented by every benchmark metric."""

    @property
    def name(self) -> str:
        ...

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        ...


def require_records(records: Sequence[EvaluationRecord]) -> tuple[EvaluationRecord, ...]:
    """Freeze and validate a non-empty record sequence."""

    frozen = tuple(records)
    if not frozen:
        raise ValueError("At least one evaluation record is required")
    return frozen


__all__ = [
    "EvaluationRecord",
    "Metric",
    "MetricResult",
    "require_records",
]
