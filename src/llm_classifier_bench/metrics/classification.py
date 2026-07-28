"""Predictive-performance metrics for single-label multiclass classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .base import EvaluationRecord, MetricResult, require_records


def accuracy_score(records: Sequence[EvaluationRecord]) -> float:
    frozen = require_records(records)
    return sum(record.gold_label == record.predicted_label for record in frozen) / len(frozen)


def macro_f1_score(
    records: Sequence[EvaluationRecord],
    *,
    labels: Sequence[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return macro-F1 and the per-class F1 values used to compute it.

    When ``labels`` is omitted, the class set is the union of gold and predicted
    labels. A runner may pass the full dataset label space later if it wants
    absent classes to contribute zero to the macro average.
    """

    frozen = require_records(records)
    class_names = tuple(labels) if labels is not None else tuple(
        sorted(
            {record.gold_label for record in frozen}
            | {record.predicted_label for record in frozen}
        )
    )
    if not class_names:
        raise ValueError("At least one class label is required")
    if len(class_names) != len(set(class_names)):
        raise ValueError("labels must not contain duplicates")

    per_class: dict[str, float] = {}
    for label in class_names:
        true_positive = sum(
            record.gold_label == label and record.predicted_label == label
            for record in frozen
        )
        false_positive = sum(
            record.gold_label != label and record.predicted_label == label
            for record in frozen
        )
        false_negative = sum(
            record.gold_label == label and record.predicted_label != label
            for record in frozen
        )

        denominator = (2 * true_positive) + false_positive + false_negative
        per_class[label] = (
            (2 * true_positive) / denominator if denominator else 0.0
        )

    return sum(per_class.values()) / len(per_class), per_class


@dataclass(frozen=True, slots=True)
class AccuracyMetric:
    name: str = "accuracy"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        return MetricResult(
            name=self.name,
            value=accuracy_score(frozen),
            metadata={"n_samples": len(frozen)},
        )


@dataclass(frozen=True, slots=True)
class MacroF1Metric:
    labels: tuple[str, ...] | None = None
    name: str = "macro_f1"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        value, per_class = macro_f1_score(frozen, labels=self.labels)
        return MetricResult(
            name=self.name,
            value=value,
            metadata={
                "n_samples": len(frozen),
                "labels": list(per_class),
                "per_class_f1": per_class,
            },
        )


__all__ = [
    "AccuracyMetric",
    "MacroF1Metric",
    "accuracy_score",
    "macro_f1_score",
]
